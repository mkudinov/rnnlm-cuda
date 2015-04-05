#include "vocabulary.h"

#define HASH_MULT 237
#define OOV_CODE -1
namespace RNNLM
{

Vocabulary::Vocabulary()
{
    m_maxSize = 100;
    m_size = 0;
    m_words = (struct vocab_word *)calloc(m_maxSize, sizeof(struct vocab_word));
    m_hashSize = 100000000;
    m_hash=(int *)calloc(m_hashSize, sizeof(int));
}

Vocabulary::~Vocabulary()
{
    free(m_words);
    free(m_hash);
}

int Vocabulary::getWordHash_(char *word)
{
    unsigned int hash=0;

    for (size_t a = 0; a < strlen(word); a++) hash = hash * HASH_MULT + word[a];

    hash=hash % m_hashSize;
    return hash;
}

int Vocabulary::search(char *word)
{
    auto hash=getWordHash_(word);

    if (m_hash[hash] == -1) return -1;
    if (!strcmp(word, m_words[m_hash[hash]].word)) return m_hash[hash];

    for (int a = 0; a < m_size; a++)
    {
        if (!strcmp(word, m_words[a].word)) {
            m_hash[hash] = a;
            return a;
        }
    }

    return OOV_CODE;	//return OOV if not found
}

int Vocabulary::addWord(char *word)
{
    strcpy(m_words[m_size].word, word);
    m_words[m_size].cn=0;
    m_size++;

    if (m_size + 2 >= m_maxSize) {        //reallocate memory if needed
        m_maxSize += 100;
        m_words=(struct vocab_word *)realloc(m_words, m_maxSize * sizeof(struct vocab_word));
    }

    auto hash = getWordHash_(word);
    m_hash[hash] = m_size - 1;

    return m_size-1;
}

void Vocabulary::sort()
{
    int max;
    vocab_word swap;

    for (int a = 1; a < m_size; a++)
    {
        max = a;
        for (int b = a + 1; b < m_size; b++)
            if (m_words[max].cn < m_words[b].cn)
            {
                max=b;
            }

        swap=m_words[max];
        m_words[max] = m_words[a];
        m_words[a] = swap;
    }
}

void Vocabulary::clearHash()
{
    m_size=0;
    for (int a = 0; a < m_hashSize; a++)
        m_hash[a] =- 1;
}

void Vocabulary::printIntoFile_(char* str) {
    char print_file[MAX_STRING];
    strcpy(print_file, str);
    FILE* fw = fopen(print_file, "w");
    fprintf (fw, "%d\n", m_size);
    for (int a = 0; a < m_size; ++a) {
        fprintf(fw, "%s %d\n", m_words[a].word, m_words[a].cn);
    }
    fclose(fw);
}

int Vocabulary::initFromFile(const std::string& i_fileName)   //assumes that vocabulary is empty
{
    char word[MAX_STRING];
    FILE *fin;
    fin=fopen(i_fileName.c_str(), "rb");
    clearHash();
    m_size=0;
    addWord((char *)"</s>");

    int train_size = 0;

    while (1)
    {
        InputSequence::readWord(word, fin);
        if (feof(fin)) break;
        train_size++;
        auto i = search(word);

        if (i == -1)
        {
            auto a = addWord(word);
            m_words[a].cn = 1;
        }
        else
            m_words[i].cn++;
    }

    sort();
    printf("Vocab size: %d\n", m_size);
    printf("Words in train file: %d\n", train_size);
    fclose(fin);
    return train_size;
}

Vocabulary::Vocabulary(Vocabulary&& i_rhs)
{
    m_words = i_rhs.words();
    m_hash = i_rhs.hash();
    m_size = i_rhs.size();
    m_maxSize = i_rhs.maxSize();
    m_hashSize = i_rhs.hashSize();
    i_rhs.release();
}

Vocabulary& Vocabulary::operator=(Vocabulary&& i_rhs)
{
    m_words = i_rhs.words();
    m_hash = i_rhs.hash();
    m_size = i_rhs.size();
    m_maxSize = i_rhs.maxSize();
    m_hashSize = i_rhs.hashSize();
    i_rhs.release();
    return *this;
}

void Vocabulary::readFromFile(FILE *fi)
{
    int vocab_size;
    m_size = 0;
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &vocab_size);
    //read normal vocabulary
    if (m_maxSize < vocab_size)
    {
        if (m_words != NULL)
        {
            free(m_words);
        }
        m_maxSize = vocab_size + 1000;
        m_words = (struct vocab_word *)calloc(m_maxSize, sizeof(struct vocab_word));    //initialize memory for vocabulary
    }

    for (int a = 0; a < vocab_size; a++)
    {
        int b;
        fscanf(fi, "%d%d", &b, &m_words[a].cn);
        InputSequence::readWord(m_words[a].word, fi);
        fscanf(fi, "%d", &b);
        m_size++;
    }
}

void Vocabulary::writeToFile(FILE *fo)
{
    fprintf(fo, "Vocabulary\n");
    fprintf(fo, "vocabulary size: %d\n", m_size);
    for (int a=0; a<m_size; a++) fprintf(fo, "%6d\t%10d\t%s\n", a, m_words[a].cn, m_words[a].word);
    fprintf(fo, "\n");
}

int InputSequence::readWordIndex_()
{
    char word[MAX_STRING];
    readWord_(word);
    if (feof(m_file)) return OOV_CODE;
    m_wordsRead++;
    return m_vocab->search(word);
}

void InputSequence::readWord_(char *word) const
{
    InputSequence::readWord(word, m_file);
}

void InputSequence::readWord(char *word, FILE* fi)
{
    int a=0, ch;

    while (!feof(fi))
    {
        ch=fgetc(fi);

        if (ch==13) continue;

        if ((ch==' ') || (ch=='\t') || (ch=='\n'))
        {
            if (a>0)
            {
                if (ch=='\n') ungetc(ch, fi);
                break;
            }

            if (ch=='\n')
            {
                strcpy(word, (char *)"</s>");
                return;
            }
            else continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) {
            //printf("Too long word found!\n");   //truncate too long words
            a--;
        }
    }
    word[a]=0;
}

bool InputSequence::end() const
{
    return feof(m_file);
}

void InputSequence::goToPosition(int i_position)
{
    if(feof(m_file) || i_position == 0)
    {
        fclose(m_file);
        m_file = fopen(m_fileName.c_str(), "rb");
        m_wordsRead = 0;
    }
    if (i_position>0)
        for (int a=0; a<i_position; a++)
        {
            readWordIndex_();	//this will skip words that were already learned if the training was interrupted
            m_wordsRead++;
        }
}

int InputSequence::next()
{
    return readWordIndex_();
}

}
