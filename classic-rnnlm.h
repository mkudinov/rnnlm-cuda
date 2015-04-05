#ifndef CLASSICRNNLM_H
#define CLASSICRNNLM_H

#include <string.h>
#include "common.h"
#include "cuda-frontend.h"

namespace RNNLM
{
struct ModelOptions
{
    ModelOptions(int i_layer1_size, int i_bptt, int i_bptt_block, bool i_independent = false) :
        independent(i_independent)
      , bptt(i_bptt)
      , bptt_block(i_bptt_block)
      , layer1_size(i_layer1_size)
    {}

    bool independent;
    int bptt;
    int bptt_block;
    int layer1_size;
};

class ClassicRnnlm
{
public:
    ClassicRnnlm() : m_logProb(double(0)) {}
    void saveNet();
    void saveWeights();			//saves current weights and unit activations
    void restoreWeights();
    void readFromFile(FILE *fi, FileTypeEnum filetype);
    void netFlush();
    void clearMemory();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)

    void computeNet(int last_word, int word);
    void learnNet(int last_word, int word, double alpha, double beta, int counter);
    void copyHiddenLayerToInput();
    bool independent() const {return m_independent;}
    void initNet(int i_vocabSize, const ModelOptions& i_options);
    double logProb() {return double(m_logProb);}
    void resetLogProb() { m_logProb = 0;}
    void writeToFile(FILE *fo, FileTypeEnum filetype);

private:
    CudaValue<double> m_logProb;

    int m_vocabSize;
    int layer1_size;

    Layer neu0;
    Layer neu1;
    Layer neu2;

    Matrix syn0v;		//weights between input and hidden layer
    Matrix syn0h;
    Matrix syn0m;
    Matrix syn1;		//weights between hidden and output layer (or hidden and compression if compression>0)

    //backup used in training:
    Layer neu0b;
    Layer neu1b;
    Layer neu2b;

    Matrix syn0vb;
    Matrix syn0hb;
    Matrix syn1b;

    //backup used in n-bset rescoring:
    Layer neu1b2;

    bool m_independent;
    int bptt;
    int bptt_block;
    std::vector<int> bptt_history;
    std::vector<Layer> bptt_hidden;
    Matrix bptt_syn0h;
    Matrix bptt_syn0v;
    Matrix bptt_syn0m;

    void computeRecurrentLayer_(int i_wordIndex);
    void computeOutputLayer_();
    void computeErrorOnOutput_(int i_trueWord);
    void applyGradient_(double i_lr, const Vector& i_next, const Vector& i_prev, Matrix& i_mat, double beta);
    void applyGradient_(double i_lr, const Vector& i_next, int i_column, Matrix& i_mat, double beta);
    void computeErrorOnPrevious_(const Layer& i_nextLayer, Matrix& i_synMat, Layer& i_prevLayer);
    void addGradient_(const Matrix& i_update, Matrix& o_mat , double beta2);
    void applyGradient_(const Matrix& i_update, int i_updateColumn, Matrix& io_target, int i_targetColumn, double i_beta);
    void updateBptt_(int last_word);
    void makeBptt_(int word, double alpha, double beta, int counter);
    void initNet_();
    void incremetLogProbByWordLP_(int word);

            //restores current weights and unit activations from backup copy
    void saveContext();
    void restoreContext();
    void saveContext2();
    void restoreContext2();

};

}
#endif // CLASSICRNNLM_H
