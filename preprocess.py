class Library:
    def __init__(self):
        self.name = "Dataset"
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)


        print(len(keep_words), len(self.word2index), len(keep_words), len(self.word2index))
        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)



def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )



def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s



def readVocs(datafile):
    pairs = []
    df = pd.read_parquet(datafile)
    questions = df['question'].tolist()
    responses = df['response'].tolist()
    for question, response in zip(questions, responses):
      question = normalizeString(question)
      response = normalizeString(response)
      pair = [question, response]
      pairs.append(pair)
    return pairs



def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH



def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]



def loadPrepareData(datafile, save_dir):
    libra = Library()
    pairs = readVocs(datafile)
    pairs = filterPairs(pairs)
    for pair in pairs:
        libra.addSentence(pair[0])
        libra.addSentence(pair[1])
    return libra, pairs



def trimRareWords(libra, pairs, MIN_COUNT):
    libra.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in libra.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in libra.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs



def indexesFromSentence(libra, sentence):
    return [libra.word2index[word] for word in sentence.split(' ') if word in libra.word2index] + [EOS_token]



def zeroPadding(l, fillvalue=PAD_token):
    padded_list = []
    for sequence in l:
        padded_sequence = list(sequence) + [fillvalue] * (MAX_LENGTH - len(sequence))
        padded_list.append(padded_sequence)
    return padded_list



def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar



# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar



def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp = inputVar(input_batch, voc)
    output = outputVar(output_batch, voc)
    return inp, output



parquet_path = "/content/drive/MyDrive/movie-corpus/movie-corpus/0000.parquet"
libra, pairs = loadPrepareData(parquet_path, save_dir)
pairs = trimRareWords(libra, pairs, MIN_COUNT)
