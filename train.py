def train(input_variable, target_variable, vocab_size, model, clip, max_length=MAX_LENGTH):

    model.encoder_optimizer.zero_grad()
    model.decoder_optimizer.zero_grad()

    t_loss = 0
    t_accuracy = 0
    n_totals = 0

    for t in range(max_length):
        decoder_output = model(input_variable)
        loss = F.cross_entropy(decoder_output.view(-1, vocab_size), target_variable.reshape(-1), ignore_index=0)
        mask = (target_variable != 0).float()
        loss = (loss * mask).mean()
        accuracy_v = accuracy(target_variable, decoder_output)
        t_loss += loss
        t_accuracy += accuracy_v.item()
        n_totals += 1

    loss.backward()

    model.encoder_optimizer.step()
    model.decoder_optimizer.step()

    model.encoder_scheduler.step()
    model.decoder_scheduler.step()

    _ = nn.utils.clip_grad_norm_(model.encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)

    return t_loss / n_totals, t_accuracy / n_totals



def trainIters(model_name, libra, pairs, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, loadFilename, vocab_size, model):

    print("Creating the training batches...")
    training_pairs = [batch2TrainData(libra, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    start_iteration = 1
    print_loss = 0
    print_count = 0
    old = 0
    tries = 0

    if loadFilename:
        tries = model.checkpoint['time']

    print("Initializing Training...")
    print()
    for iteration in range(start_iteration, n_iteration + 1):
        training_pair = training_pairs[iteration - 1]

        input_variable, target_variable = training_pair

        input_variable = input_variable.to(device)
        target_variable = target_variable.to(device)

        input_variable = model.embedding(input_variable).to(device)

        loss, accuracy = train(input_variable, target_variable, vocab_size, model, clip)

        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}; Average accuracy: {:.4f}".format(iteration, iteration / n_iteration * 100, loss, accuracy))


        if (iteration % save_every == 0):
                    tries += save_every
                    directory = os.path.join(save_dir, model_name, '{}-{}_{}'.format(model.embedding_size, model.num_heads, model.vocab_size))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save({
                        'iteration': iteration,
                        'time': tries,
                        'mo':model,
                        'en': model.encoder.state_dict(),
                        'de': model.decoder.state_dict(),
                        'en_opt': model.encoder_optimizer.state_dict(),
                        'de_opt': model.decoder_optimizer.state_dict(),
                        'loss': loss,
                        'voc_dict': libra.__dict__,
                        'embedding': model.embedding.state_dict()
                    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
