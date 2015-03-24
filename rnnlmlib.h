///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#ifndef _RNNLMLIB_H_
#define _RNNLMLIB_H_

#define MAX_STRING 100

#include <assert.h>
#include "cuda-frontend.h"

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

#include <string.h>

struct vocab_word {
    int cn;
    char word[MAX_STRING];

    double prob;
    int class_index;
};

const int MAX_NGRAM_ORDER=20;

enum FileTypeEnum {TEXT, BINARY, COMPRESSED};		//COMPRESSED not yet implemented

class CRnnLM{
protected:
    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    char lmprob_file[MAX_STRING];
    
    int rand_seed;
    
    int debug_mode;
    
    int version;
    int filetype;
    
    int use_lmprob;
    double lambda;
    double gradient_cutoff;
    
    double dynamic;
    
    double alpha;
    double starting_alpha;
    int alpha_divide;
    double logp, llogp;
    float min_improvement;
    int iter;
    int vocab_max_size;
    int train_words;
    int train_cur_pos;
    int counter;
    
    int one_iter;
    int anti_k;
    
    double beta;
    
    int class_size;
    int **class_words;
    int *class_cn;
    int *class_max_cn;
    int old_classes;
    
    struct vocab_word *vocab;
    void sortVocab();
    int *vocab_hash;
    int vocab_hash_size;

    int vocab_size;
    int layer1_size;
    
    long long direct_size;
    int direct_order;
    int history[MAX_NGRAM_ORDER];
    
    int bptt;
    int bptt_block;
    std::vector<int> bptt_history;
    std::vector<Layer> bptt_hidden;
    Matrix bptt_syn0h;
    Matrix bptt_syn0v;
    
    int gen;

    int independent;
    
    Layer neu0;
    Layer neu1;
    Layer neu2;

    Matrix syn0v;		//weights between input and hidden layer
    Matrix syn0h;
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
    
public:

    int alpha_set, train_file_set;

    CRnnLM()		//constructor initializes variables
    {
	version=10;
	filetype=TEXT;
	
	use_lmprob=0;
	lambda=0.75;
	gradient_cutoff=15;
	dynamic=0;
    
	train_file[0]=0;
	valid_file[0]=0;
	test_file[0]=0;
	rnnlm_file[0]=0;
	
	alpha_set=0;
	train_file_set=0;
	
	alpha=0.1;
	beta=0.0000001;
	alpha_divide=0;
	logp=0;
	llogp=-100000000;
	iter=0;
	
	min_improvement=1.003;
	
	train_words=0;
	train_cur_pos=0;
	vocab_max_size=100;
	vocab_size=0;
	vocab=(struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	
	layer1_size=30;
	
	direct_size=0;
	direct_order=0;
	
	bptt=0;
	bptt_block=10;
	
	gen=0;

	independent=0;
	
	rand_seed=1;
	
	class_size=100;
	old_classes=0;
	
	one_iter=0;
	
	debug_mode=1;
	srand(rand_seed);
	
	vocab_hash_size=100000000;
	vocab_hash=(int *)calloc(vocab_hash_size, sizeof(int));
    }
    
    ~CRnnLM()		//destructor, deallocates memory
    {
        int i;
        if (vocab==NULL)
        {
            for (i=0; i<class_size; i++) free(class_words[i]);
            free(class_max_cn);
            free(class_cn);
            free(class_words);

            free(vocab);
            free(vocab_hash);
            //todo: free bptt variables too
        }
    }
    
    double random(double min, double max);

    void setTrainFile(char *str);
    void setValidFile(char *str);
    void setTestFile(char *str);
    void setRnnLMFile(char *str);
    void setLMProbFile(char *str) {strcpy(lmprob_file, str);}
    
    void setFileType(int newt) {filetype=newt;}
    
    void setClassSize(int newSize) {class_size=newSize;}
    void setOldClasses(int newVal) {old_classes=newVal;}
    void setLambda(double newLambda) {lambda=newLambda;}
    void setGradientCutoff(double newGradient) {gradient_cutoff=newGradient;}
    void setDynamic(double newD) {dynamic=newD;}
    void setGen(double newGen) {gen=newGen;}
    void setIndependent(int newVal) {independent=newVal;}
    
    void setLearningRate(double newAlpha) {alpha=newAlpha;}
    void setRegularization(double newBeta) {beta=newBeta;}
    void setMinImprovement(double newMinImprovement) {min_improvement=newMinImprovement;}
    void setHiddenLayerSize(int newsize) {layer1_size=newsize;}
    void setDirectSize(long long newsize) {direct_size=newsize;}
    void setDirectOrder(int newsize) {direct_order=newsize;}
    void setBPTT(int newval) {bptt=newval;}
    void setBPTTBlock(int newval) {bptt_block=newval;}
    void setRandSeed(int newSeed) {rand_seed=newSeed; srand(rand_seed);}
    void setDebugMode(int newDebug) {debug_mode=newDebug;}
    void setAntiKasparek(int newAnti) {anti_k=newAnti;}
    void setOneIter(int newOneIter) {one_iter=newOneIter;}
    
    int getWordHash(char *word);
    void readWord(char *word, FILE *fin);
    int searchVocab(char *word);
    int readWordIndex(FILE *fin);
    int addWordToVocab(char *word);
    void learnVocabFromTrainFile();		//train_file will be used to construct vocabulary
    void PrintVocabIntoFile(char* str);

    void saveWeights();			//saves current weights and unit activations
    void restoreWeights();		//restores current weights and unit activations from backup copy
    //void saveWeights2();		//allows 2. copy to be stored, useful for dynamic rescoring of nbest lists
    //void restoreWeights2();		
    void saveContext();
    void restoreContext();
    void saveContext2();
    void restoreContext2();
    void initNet();
    void saveNet();
    void goToDelimiter(int delim, FILE *fi);
    void restoreNet();
    void netFlush();
    void netReset();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)
    
    void computeNet(int last_word, int word);
    void learnNet(int last_word, int word);
    void copyHiddenLayerToInput();
    void trainNet();
    void useLMProb(int use) {use_lmprob=use;}
    void testNet();
    void testNbest();
    void testGen();
    void computeRecurrentLayer_(int i_wordIndex);
    void computeOutputLayer_();
    void computeErrorOnOutput_(int i_trueWord);
    void applyGradient_(double i_lr, const Vector& i_next, const Vector& i_prev, Matrix& i_mat, double beta);
    void applyGradient_(double i_lr, const Vector& i_next, int i_column, Matrix& i_mat, double beta);
    void computeErrorOnPrevious_(const Layer& i_nextLayer, Matrix& i_synMat, Layer& i_prevLayer);
    void addGradient_(const Matrix& i_update, Matrix& o_mat , double beta2);
    void applyGradient_(const Matrix& i_update, int i_updateColumn, Matrix& io_target, int i_targetColumn, double i_beta);
};

#endif
