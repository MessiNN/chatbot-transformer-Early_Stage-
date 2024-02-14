def evaluate(sentence, model, libra, embedding_size):
    # Tokenize the sentence
    tokenized_sentence = [SOS_token] + [libra.word2index[word] for word in sentence.split()] + [EOS_token]
    # Convert to tensor and add batch dimension
    sentence_tensor = torch.tensor(tokenized_sentence, dtype=torch.long).unsqueeze(0)

    # Initialize output tensor with start token
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])

    for i in range(MAX_LENGTH):
        with torch.no_grad():
            predictions = model(sentence_tensor)

        # Select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        # Return the result if the predicted_id is equal to the end token
        if predicted_id.item() == EOS_token:
            break

        # Concatenate the predicted_id to the output
        output = torch.cat([decoder_input, predicted_id], axis=-1)

    return output.squeeze(0)


def predict(input_sentence, model, libra, embedding_size):
    prediction = evaluate(input_sentence, model, libra, embedding_size)
    predicted_sentence = [libra.index2word[index.item()] for index in prediction if index.item() < libra.num_words]
    return predicted_sentence


def evaluateInput(model, libra, embedding_size):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('User > ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = normalizeString(input_sentence)
            output_words = predict(input_sentence, model, libra, embedding_size)
            print('Cleopatra:', ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")
