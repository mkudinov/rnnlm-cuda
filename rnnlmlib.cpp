///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"
#include <sys/time.h>
#include <iostream>

namespace RNNLM
{

CRnnLM::CRnnLM()		//constructor initializes variablesp
{
    filetype=TEXT;
    gradient_cutoff=15;
    alpha_set=0;
    train_file_set=0;
    alpha=0.1;
    beta=0.0000001;
    alpha_divide=0;
    iter=0;
    min_improvement=1.003;
    int rand_seed=1;
    debug_mode=1;
    srand(rand_seed);
}

void CRnnLM::initTraining_(char *train_file, char *valid_file, char *snapshot_file, const ModelOptions& i_options)
{
    FILE *fi;

    m_trainWords = 0;

    fi = fopen(snapshot_file, "rb");
    if (fi!=NULL)
    {
        fclose(fi);
        printf("Restoring network from file to continue training...\n");
        restoreFromSnapshot_(snapshot_file, m_vocab, m_model);
    }
    else
    {
        m_trainWords = m_vocab.initFromFile(train_file);
        m_vocabSize = m_vocab.size();
        m_model.initNet(m_vocabSize, i_options);
        iter=0;
    }

    m_trainSource = InputSequence(train_file, &m_vocab);
    m_validSource = InputSequence(valid_file, &m_vocab);
}

std::tuple<double, clock_t, int> CRnnLM::learningPhase_()
{
    clock_t start, now;
    int last_word=0;
    int counter = 0;
    m_model.netFlush();
    start=clock();
    //for unknown reason mikolov counts all words including OOV and EOF when calculating LogProb on train.

    m_trainSource.goToPosition(0);
    while (1)
    {
        counter++;

        if ((counter%10000)==0) if ((debug_mode>1)) {
        now=clock();
        if (m_trainWords>0)
            printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 13, iter, alpha, -m_model.logProb()/log10(2)/counter, counter/(double)m_trainWords*100, counter/((double)(now-start)/1000000.0));
        else
            printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 13, iter, alpha, -m_model.logProb()/log10(2)/counter, counter/1000);
        fflush(stdout);
        }

        int word = m_trainSource.next(); //readWordIndex(fi);     //read next word
        m_model.computeNet(last_word, word); //compute probability distribution
        if(m_trainSource.end()) break;
        m_model.learnNet(last_word, word, alpha, beta, counter); //update model
        m_model.copyHiddenLayerToInput(); //update hidden layer
        last_word=word;
        if (m_model.independent() && (word==0)) m_model.clearMemory();

    }
      now=clock();

      std::tuple<double, clock_t, int> trainResult;
      std::get<0>(trainResult) = m_model.logProb();
      std::get<1>(trainResult) = now - start;
      std::get<2>(trainResult) = counter;
      m_model.resetLogProb();
      return trainResult;
}

std::tuple<double, int> CRnnLM::validationPhase_()
{
    m_model.netFlush();
    int wordcn=0;
    int last_word = 0;

    m_validSource.goToPosition(0);
    while (1)
    {
        int word = m_validSource.next();
        m_model.computeNet(last_word, word);      //compute probability distribution
        if(m_validSource.end()) break;      //end of file: report LOGP, PPL

        if (word!=-1)
        {
            wordcn++;
        }

        m_model.copyHiddenLayerToInput();
        last_word=word;

        if (m_model.independent() && (word==0)) m_model.clearMemory();
    }

    std::tuple<double, int> validResult;
    std::get<0>(validResult) = m_model.logProb();
    std::get<1>(validResult) = wordcn;
    m_model.resetLogProb();
    return validResult;
}

void CRnnLM::trainNet(char *train_file, char *valid_file, char *snapshot_file, const ModelOptions& i_options)
{
    printf("Starting training using file %s\n", train_file);
    starting_alpha=alpha;
    initTraining_(train_file, valid_file, snapshot_file, i_options);

    double logp=0;
    double llogp=-100000000;

    while (1)
    {
        printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
        fflush(stdout);
        double train_logp;
        clock_t train_time;
        int sampleSize;
        std::tie(train_logp, train_time, sampleSize) = learningPhase_();
        printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -train_logp/log10(2)/sampleSize, m_trainWords/((double)(train_time)/1000000.0));
        std::tie(logp, sampleSize) = validationPhase_();
        printf("VALID entropy: %.4f\n", -logp/log10(2)/sampleSize);

        if (logp<llogp)
        {
            m_model.restoreWeights();
        }
        else
        {
            m_model.saveWeights();
        }

        if (logp*min_improvement<llogp)
        {
            if (alpha_divide==0)
                alpha_divide=1;
            else
            {
                saveSnapshot_(train_file, valid_file, snapshot_file);
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        saveSnapshot_(train_file, valid_file, snapshot_file);
        iter++;
    }
}

void CRnnLM::saveSnapshot_(const std::string& i_trainFileName, const std::string& i_validFileName, const std::string& i_snapshotFileName)       //will save the whole network structure
{
    FILE *fo;
    char str[1000];

    sprintf(str, "%s.temp", i_snapshotFileName.c_str());

    fo=fopen(str, "wb");
    if (fo==NULL) {
        printf("Cannot create file %s\n", i_snapshotFileName.c_str());
        exit(1);
    }

    Snapshot snapshot;
    snapshot.alpha = alpha;
    snapshot.alpha_divide = alpha_divide;
    snapshot.starting_alpha = starting_alpha;
    snapshot.beta = beta;
    snapshot.filetype = filetype;
    snapshot.gradient_cutoff = gradient_cutoff;
    snapshot.iter = iter;
    snapshot.train_file = i_trainFileName.c_str();
    snapshot.valid_file = i_validFileName.c_str();
    snapshot.train_words = m_trainWords;

    snapshot.writeToFile(fo);
    m_vocab.writeToFile(fo);
    m_model.writeToFile(fo, filetype);

    fclose(fo);
    rename(str, i_snapshotFileName.c_str());
}

void CRnnLM::restoreFromSnapshot_(char *i_snapshot_file, Vocabulary& o_vocab, ClassicRnnlm& o_model)    //will read whole network structure
{
    FILE *fi;

    fi=fopen(i_snapshot_file, "rb");
    if (fi==NULL)
    {
        printf("ERROR: model file '%s' not found!\n", i_snapshot_file);
        exit(1);
    }

    Snapshot snapshot;
    snapshot.readFromFile(fi);
    o_vocab.readFromFile(fi);
    m_vocabSize = o_vocab.size();
    o_model.readFromFile(fi, filetype);
    fclose(fi);
}

void Snapshot::readFromFile(FILE *fi)
{
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", train_file.c_str());
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", valid_file.c_str());
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    int train_words;
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &starting_alpha);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &alpha);

    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
}

void Snapshot::writeToFile(FILE *fo)
{
    fprintf(fo, "RNNLM model file\n");
    fprintf(fo, "file format: %d\n\n", filetype);
    fprintf(fo, "training data file: %s\n", train_file.c_str());
    fprintf(fo, "validation data file: %s\n\n", valid_file.c_str());
    fprintf(fo, "number of finished iterations: %d\n", iter);
    fprintf(fo, "# of training words: %d\n", train_words);
    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");
}


}
