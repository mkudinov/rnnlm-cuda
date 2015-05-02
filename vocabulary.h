#ifndef VOCABULARY_H
#define VOCABULARY_H

#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <tuple>
#include "common.h"

namespace RNNLM
{
class InputSequence;

struct vocab_word {
    int cn;
    char word[MAX_STRING];
    double prob;
};

class Vocabulary
{
public:
    Vocabulary();
    ~Vocabulary();
    Vocabulary(const Vocabulary& i_rhs) = delete;
    Vocabulary& operator=(const Vocabulary& i_rhs) = delete;
    Vocabulary(Vocabulary&& i_rhs);
    Vocabulary& operator=(Vocabulary&& i_rhs);

    int addWord(char *i_word);
    int search(char *i_word);
    void clearHash();
    void sort();
    void readFromFile(FILE *fi);
    void writeToFile(FILE *fo);
    vocab_word& operator[](size_t i) { return m_words[i];}


    //int initFromFile(const std::string& i_fileName);

    vocab_word *words() const {return m_words;}
    int hashSize() const {return m_hashSize;}
    int maxSize() const {return m_maxSize;}
    int size() const {return m_size;}
    int *hash() const {return m_hash;}

    void release()
    {
        m_words = NULL;
        m_hash = NULL;
        m_hashSize = 0;
        m_maxSize = 0;
        m_size = 0;
    }

    void newWord(int a)
    {
        m_words[a].cn = 1;
    }

    void updateWordCount(int i)
    {
        m_words[i].cn++;
    }

private:
    void printIntoFile_(char* str);
    int getWordHash_(char *i_word);
    int m_hashSize;
    int m_maxSize;
    int m_size;
    int *m_hash;
    vocab_word *m_words;
};

class InputSequence
{
public:
    InputSequence() : m_fileName("")
                    , m_wordsRead(0){}
    InputSequence(const std::string& i_fileName, Vocabulary *i_vocab) :
         m_fileName(i_fileName)
       , m_vocab(i_vocab)
       , m_wordsRead(0)
    {
        m_file = fopen(m_fileName.c_str(), "rb");
        if(!m_file)
        {
            printf("File %s does not exit.", m_fileName.c_str());
            exit(1);
        }
        readWordIndex_();
        if(feof(m_file))
            exit(1);
    }
    ~InputSequence()
    {
        if(m_file)
            fclose(m_file);
    }
    InputSequence& operator=(const InputSequence& ) = delete;
    InputSequence& operator=(InputSequence&& i_rhs)
    {
        m_file = i_rhs.file();
        m_fileName = i_rhs.fileName();
        m_vocab = i_rhs.vocab();
        m_wordsRead = 0;
        i_rhs.release();
        m_file = fopen(m_fileName.c_str(), "rb");
        return *this;
    }

    std::string fileName() const {return m_fileName;}
    FILE* file() const { return m_file; }
    Vocabulary *vocab() const { return m_vocab;}
  //  size_t nWordsRead() const {return m_wordsRead;}
    void goToPosition(int i_position);
    int next();
    bool end() const;
    static void readWord(char *word, FILE* fi);

    void release()
    {
        m_file  = NULL;
        m_vocab = NULL;
        m_wordsRead = 0;
        m_fileName = "";
    }

private:
    void readWord_(char *word) const;
    int readWordIndex_();
    std::string m_fileName;
    Vocabulary *m_vocab;
    size_t m_wordsRead;
    FILE *m_file;
};

class InputPairSequence
{
public:
    InputPairSequence() : m_fileName("")
                    , m_wordsRead(0){}
    InputPairSequence(const std::string& i_fileName, Vocabulary *i_vocab, Vocabulary *i_morphInventory) :
         m_fileName(i_fileName)
       , m_vocab(i_vocab)
       , m_morphInventory(i_morphInventory)
       , m_wordsRead(0)
    {
        m_file = fopen(m_fileName.c_str(), "rb");
        if(!m_file)
        {
            printf("File %s does not exit.", m_fileName.c_str());
            exit(1);
        }
        readWordIndex_();
        if(feof(m_file))
            exit(1);
    }
    ~InputPairSequence()
    {
        if(m_file)
            fclose(m_file);
    }
    InputPairSequence& operator=(const InputPairSequence& ) = delete;
    InputPairSequence& operator=(InputPairSequence&& i_rhs)
    {
        m_file = i_rhs.file();
        m_fileName = i_rhs.fileName();
        m_vocab = i_rhs.m_vocab;
        m_morphInventory = i_rhs.m_morphInventory;
        m_wordsRead = 0;
        i_rhs.release();
        m_file = fopen(m_fileName.c_str(), "rb");
        return *this;
    }

    std::string fileName() const {return m_fileName;}
    FILE* file() const { return m_file; }
    Vocabulary *vocab() const { return m_vocab;}
    Vocabulary *morphology() const {return m_morphInventory;}
  //  size_t nWordsRead() const {return m_wordsRead;}
    void goToPosition(int i_position);
    std::tuple<int,int> next();
    bool end() const;
    static void readWord(char *word, char *morph, FILE* fi);
    static std::tuple<int, Vocabulary, Vocabulary> initFromFile(const std::string& i_fileName);
    void release()
    {
        m_file  = NULL;
        m_vocab = NULL;
        m_morphInventory = NULL;
        m_wordsRead = 0;
        m_fileName = "";
    }

private:
    void readWord_(char *word, char *morph) const;
    std::tuple<int,int> readWordIndex_();
    std::string m_fileName;
    Vocabulary *m_vocab;
    Vocabulary *m_morphInventory;
    size_t m_wordsRead;
    FILE *m_file;
};

class ENotInitiated
{
public:
    ENotInitiated()  {printf("EXCEPTION: Source is not initiated.");}
};

}
#endif // VOCABULARY_H
