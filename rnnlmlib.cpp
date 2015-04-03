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

CRnnLM::CRnnLM()		//constructor initializes variables
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

double CRnnLM::validationPhase_()
{
    m_model.netFlush();
  //  int wordcn=0;
    int last_word = 0;

    m_validSource.goToPosition(0);
    while (1)
    {
        int word = m_validSource.next();
        m_model.computeNet(last_word, word);      //compute probability distribution
        if(m_validSource.end()) break;      //end of file: report LOGP, PPL

//        if (word!=-1)
//        {
//            logp += log10(m_model.wordScore(word));
//            wordcn++;
//        }

        m_model.copyHiddenLayerToInput();
        last_word=word;

        if (m_model.independent() && (word==0)) m_model.clearMemory();
    }

    double logp = m_model.logProb();
    m_model.resetLogProb();
    return logp;
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
        logp = validationPhase_();
        printf("VALID entropy: %.4f\n", -logp/log10(2)/m_validSource.nWordsRead());

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
                saveSnapshot_();
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        saveSnapshot_();
        iter++;
    }
}

void CRnnLM::saveSnapshot_()       //will save the whole network structure
{
//    FILE *fo;
//    int a, b;
//    char str[1000];
//    float fl;

//    sprintf(str, "%s.temp", rnnlm_file);

//    fo=fopen(str, "wb");
//    if (fo==NULL) {
//        printf("Cannot create file %s\n", rnnlm_file);
//        exit(1);
//    }
//    //fprintf(fo, "version: %d\n", version);
//    fprintf(fo, "file format: %d\n\n", filetype);

//    fprintf(fo, "training data file: %s\n", train_file);
//    fprintf(fo, "validation data file: %s\n\n", valid_file);

//    fprintf(fo, "last probability of validation data: %f\n", llogp);
//    fprintf(fo, "number of finished iterations: %d\n", iter);

//    fprintf(fo, "current position in training data: %d\n", train_cur_pos);
//    fprintf(fo, "current probability of training data: %f\n", logp);
//    fprintf(fo, "save after processing # words: %d\n", 0);
//    fprintf(fo, "# of training words: %d\n", train_words);

//    fprintf(fo, "input layer size: %d\n", vocab_size+layer1_size);
//    fprintf(fo, "hidden layer size: %d\n", layer1_size);
//    fprintf(fo, "compression layer size: %d\n", 0);
//    fprintf(fo, "output layer size: %d\n", vocab_size + 1);

//    fprintf(fo, "direct connections: %lld\n", 0);
//    fprintf(fo, "direct order: %d\n", 0);

//    fprintf(fo, "bptt: %d\n", bptt);
//    fprintf(fo, "bptt block: %d\n", bptt_block);

//    fprintf(fo, "vocabulary size: %d\n", vocab_size);
//    fprintf(fo, "class size: %d\n", class_size);

//    fprintf(fo, "old classes: %d\n", old_classes);
//    fprintf(fo, "independent sentences mode: %d\n", independent);

//    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
//    fprintf(fo, "current learning rate: %f\n", alpha);
//    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
//    fprintf(fo, "\n");

//    fprintf(fo, "\nVocabulary:\n");
//    for (a=0; a<vocab_size; a++) fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, vocab[a].cn, vocab[a].word, vocab[a].class_index);

//    neu1.ac.prepareToSave();
//    if (filetype==TEXT)
//    {
//        fprintf(fo, "\nHidden layer activation:\n");
//        for (a=0; a<layer1_size; a++) fprintf(fo, "%.4f\n", neu1.ac[a]);
//    }
//    if (filetype==BINARY)
//    {
//        for (a=0; a<layer1_size; a++)
//        {
//            fl=neu1.ac[a];
//            fwrite(&fl, 4, 1, fo);
//        }
//    }
//    //////////
//    syn0v.prepareToSave();
//    syn0h.prepareToSave();
//    syn1.prepareToSave();

////    syn0v.print();
////    syn0h.print();
////    syn1.print();
////    neu1.ac.print();

//    if (filetype==TEXT)
//    {
//        fprintf(fo, "\nWeights 0->1:\n");
//        for (b=0; b<layer1_size; b++)
//        {
//            for (a=0; a<vocab_size; a++)
//            {
//                fprintf(fo, "%.4f\n", syn0v.getElement(b,a));
//            }
//            for (a=0; a<layer1_size; a++)
//            {
//                fprintf(fo, "%.4f\n", syn0h.getElement(b,a));
//            }
//        }
//    }
//    if (filetype==BINARY)
//    {
//        for (b=0; b<layer1_size; b++)
//        {
//            for (a=0; a<vocab_size; a++)
//            {
//                fl=syn0v.getElement(b,a);
//                fwrite(&fl, 4, 1, fo);
//            }
//            for (a=0; a<layer1_size; a++)
//            {
//                fl=syn0h.getElement(b,a);
//                fwrite(&fl, 4, 1, fo);
//            }
//        }
//    }
//    /////////
//    if (filetype==TEXT)
//    {
//        fprintf(fo, "\n\nWeights 1->2:\n");
//        for (b=0; b<vocab_size; b++)
//        {
//            for (a=0; a<layer1_size; a++)
//            {
//                fprintf(fo, "%.4f\n", syn1.getElement(b,a));
//            }
//        }
//        for (a=0; a<layer1_size; a++)
//        {
//            fprintf(fo, "%.4f\n", 1.0); //we removed classes so for compatibility with the original tool we leave here class weights
//        }

//    }
//    if (filetype==BINARY)
//    {
//        for (b=0; b<vocab_size; b++)
//        {
//            for (a=0; a<layer1_size; a++)
//            {
//                fl=syn1.getElement(b,a);
//                fwrite(&fl, 4, 1, fo);
//            }
//        }
//        for (a=0; a<layer1_size; a++)
//        {
//            fl=1.0;
//            fwrite(&fl, 4, 1, fo); //same here
//        }
//    }
//    ////////
//    if (filetype==TEXT)
//    {
//        fprintf(fo, "\nDirect connections:\n"); //for compatibility
//    }
//    ////////
//    fclose(fo);

//    rename(str, rnnlm_file);
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
    fscanf(fi, "%lf", &llogp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_cur_pos);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &logp);
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

}
