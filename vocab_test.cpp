#include <tuple>
#include <iostream>
#include "vocabulary.h"

int main()
{
    int trainWords;
    char *train_file = "/data/home/mkudinov/Data/lenta_morph_test/fake_morph";
    RNNLM::Vocabulary vocab, morphVocab;
    std::tie(trainWords,vocab,morphVocab) = RNNLM::InputPairSequence::initFromFile(train_file);
    std::cout << "Train words: " << trainWords << std::endl;

    RNNLM::InputPairSequence trainSource(train_file, &vocab, &morphVocab);
    int last_word=0, last_morph = 0;

    trainSource.goToPosition(0);
    while (1)
    {
        int word = -2, morph = -2;
        std::tie(word,morph) = trainSource.next(); //readWordIndex(fi);     //read next word
        if(trainSource.end()) break;
        std::cout << "Present lemma: " << word << "; present morph: " << morph << std::endl << "Last lemma: " << last_word << "; last morph: " << last_morph << std::endl << std::endl;
        last_word=word;
        last_morph = morph;
    }
    return 0;
}
