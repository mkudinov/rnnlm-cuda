#include "rnnlm-russian-morphology.h"

const double tau = 0.5;

namespace RNNLM
{
void RnnlmRussianMorphology::computeNet(int last_word, int last_morph, int word, int morph)
{
    computeRecurrentLayer_(last_word, last_morph);
    if (word!=-1)
    {
        computeOutputLayer_();
        incremetLogProbByWordLP_(word, morph);
    }
}

void RnnlmRussianMorphology::learnNet(int last_word, int last_morph, int word, int morph, double alpha, double beta, int counter)
{
    if (bptt>0) updateBptt_(last_word, last_morph); //shift bptt layers
    double beta2 = beta*alpha;
    if (word==-1) return;
    computeErrorOnOutput_(word, morph);
    computeErrorOnHidden_(neu2v, neu2m, syn1v, syn1m, neu1);

    if ((counter%10)==0)
    {
        applyGradient_(alpha, neu2v.er, neu1.ac, syn1v, beta2);
        applyGradient_(alpha, neu2m.er, neu1.ac, syn1m, beta2);
    }
    else
    {
        applyGradient_(alpha, neu2v.er, neu1.ac, syn1v, 0);
        applyGradient_(alpha, neu2m.er, neu1.ac, syn1m, 0);
    }

  //  syn1.print();

    if (bptt<=1)
    {
        neu1.logisticErrorActivation();
        //weight update 1->0
        if (last_word!=-1)
        {
            if ((counter%10)==0)
            {
                applyGradient_(alpha, neu1.er, last_word, syn0v, beta2);
                applyGradient_(alpha, neu1.er, last_morph, syn0m, beta2);
            }
            else
            {
                applyGradient_(alpha, neu1.er, last_word, syn0v, 0);
                applyGradient_(alpha, neu1.er, last_morph, syn0m, 0);
            }
        }
        if ((counter%10)==0)
            applyGradient_(alpha,neu1.er,neu0.ac,syn0h,beta2);
        else
            applyGradient_(alpha,neu1.er,neu0.ac,syn0h,0);
    }
    else		//BPTT
    {
        makeBptt_(word, morph, alpha, beta, counter);
    }
}

void RnnlmRussianMorphology::copyHiddenLayerToInput()
{
    neu0.ac = neu1.ac;
}

void RnnlmRussianMorphology::saveWeights()      //saves current weights and unit activations
{
    neu0b = neu0;
    neu1b = neu1;
    neu2vb = neu2v;
    neu2mb = neu2m;
    syn0vb = syn0v;
    syn0mb = syn0m;
    syn0hb = syn0h;
    syn1vb = syn1v;
    syn1mb = syn1m;
}

void RnnlmRussianMorphology::restoreWeights()      //restores current weights and unit activations from backup copy
{
    neu0 = neu0b;
    neu1 = neu1b;
    neu2v = neu2vb;
    neu2m = neu2mb;
    syn0v = syn0vb;
    syn0m = syn0mb;
    syn0h = syn0hb;
    syn1v = syn1vb;
    syn1m = syn1mb;
}

void RnnlmRussianMorphology::saveContext()		//useful for n-best list processing
{
    neu1b.ac = neu1.ac;
}

void RnnlmRussianMorphology::restoreContext()
{
    neu1.ac = neu1b.ac;
}

void RnnlmRussianMorphology::initNet(int i_vocabSize, int i_morphSize, const ModelOptions& i_options)
{
    layer1_size = i_options.layer1_size;
    bptt = i_options.bptt;
    bptt_block = i_options.bptt_block;
    m_independent = i_options.independent;
    m_vocabSize = i_vocabSize;
    m_morphologySize = i_morphSize;
    initNet_();
}

void RnnlmRussianMorphology::initNet_()
{
    neu0.setConstant(layer1_size, 0);
    neu1.setConstant(layer1_size, 0);
    neu2v.setConstant(m_vocabSize, 0);
    neu2m.setConstant(m_morphologySize, 0);

    neu0b.setConstant(layer1_size, 0);
    neu1b.setConstant(layer1_size, 0);
    neu2vb.setConstant(m_vocabSize, 0);
    neu2mb.setConstant(m_morphologySize, 0);

    syn0vb.setZero(m_vocabSize, layer1_size);
    syn0mb.setZero(m_morphologySize, layer1_size);
    syn0hb.setZero(layer1_size,layer1_size);

    syn1vb.setZero(layer1_size, m_vocabSize);
    syn1mb.setZero(layer1_size, m_morphologySize);

    double* syn0_init=(double *)calloc(layer1_size*(m_vocabSize+layer1_size), sizeof(double));
    double* syn0v_init=(double *)calloc(layer1_size*m_vocabSize, sizeof(double));
    double* syn1v_init = (double *)calloc(layer1_size*(m_vocabSize+1), sizeof(double));
    double* syn1m_init = (double *)calloc(layer1_size*(m_morphologySize+1), sizeof(double));
    double* syn0m_init=(double *)calloc(layer1_size*m_morphologySize, sizeof(double));
    double* syn0h_init=(double *)calloc(layer1_size*layer1_size, sizeof(double));

    for (int b=0; b<layer1_size; b++) for (int a=0; a<m_vocabSize+layer1_size; a++)
    {
        syn0_init[a+b*(m_vocabSize+layer1_size)]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); //init block intended to initialize the same matrices as in mikolov's implementation (for debug purposes)
    }

    for (int b=0; b<m_vocabSize+1; b++) for (int a=0; a<layer1_size; a++)
    {
        syn1v_init[a+b*layer1_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
    }

//    for (b=0; b<layer1_size; b++) for (a=0; a<morph_size; a++)
//    {
//        syn0_init[a+b*(morph_size)]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); //init block intended to initialize the same matrices as in mikolov's implementation (for debug purposes)
//    }

    for(int i = 0; i < layer1_size; i++)
    {
        for(int j = 0; j < m_vocabSize; j++)
        {
            syn0v_init[i * m_vocabSize + j] = syn0_init[j + (m_vocabSize+layer1_size) * i];
        }
    }

    for(int i = 0; i < layer1_size; i++)
    {
        for(int j = 0; j < layer1_size; j++)
        {
            syn0h_init[i * layer1_size + j] = syn0_init[m_vocabSize + j + (m_vocabSize+layer1_size) * i];
        }
    }

    free(syn0_init);

    for (int b=0; b<m_morphologySize+1; b++) for (int a=0; a<layer1_size; a++)
    {
        syn1m_init[a+b*layer1_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
    }

//    for(int i = 0; i < layer1_size; i++)
//    {
//        for(int j = 0; j < m_vocabSize; j++)
//        {
//            syn0v_init[i * m_vocabSize + j] = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
//        }
//    }

    for(int i = 0; i < layer1_size; i++)
    {
        for(int j = 0; j < m_morphologySize; j++)
        {
            syn0m_init[i * m_morphologySize+ j] = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
        }
    }

//    for(int i = 0; i < layer1_size; i++)
//    {
//        for(int j = 0; j < layer1_size; j++)
//        {
//            syn0h_init[i * layer1_size + j] = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
//        }
//    }

    syn0v.setMatrix(syn0v_init, layer1_size, m_vocabSize);
    syn0m.setMatrix(syn0m_init, layer1_size, m_morphologySize);
    syn0h.setMatrix(syn0h_init, layer1_size, layer1_size);
    syn1v.setMatrix(syn1v_init, m_vocabSize, layer1_size);
    syn1m.setMatrix(syn1m_init, m_morphologySize, layer1_size);

    free(syn1v_init);
    free(syn1m_init);
    free(syn0v_init);
    free(syn0m_init);
    free(syn0h_init);

    if (bptt>0)
    {
        bptt_history_v = std::vector<int>(bptt+bptt_block+10, -1);
        bptt_history_m = std::vector<int>(bptt+bptt_block+10, -1);

        for (size_t i=0; i < bptt + bptt_block; i++)
        {
            Layer newLayer;
            newLayer.setConstant(layer1_size, 0);
            bptt_hidden.push_back(std::move(newLayer));
        }

        bptt_syn0v.setZero(layer1_size, m_vocabSize);
        bptt_syn0m.setZero(layer1_size, m_morphologySize);
        bptt_syn0h.setZero(layer1_size, layer1_size);
    }

    saveWeights();
}

void RnnlmRussianMorphology::netFlush()   //cleans all activations and error vectors
{
    neu0.ac.setConstant(0.1);
    neu0.er.setZero();

    neu1.ac.setZero();
    neu1.er.setZero();

    neu2v.ac.setZero();
    neu2v.er.setZero();

    neu2m.ac.setZero();
    neu2m.er.setZero();

    if (bptt>0)
    {
        for (int a=0; a<bptt+bptt_block; a++) bptt_history_v[a]=0;
        for (int a=0; a<bptt+bptt_block; a++) bptt_history_m[a]=0;
    }
}

void RnnlmRussianMorphology::clearMemory()   //cleans hidden layer activation + bptt history
{
    int a;
    //it really looks like bug: neu1.ac is rewritten at the beginning of computeNet. It is lost unless it is stored in neu0.ac. Setting it to 1 (btw, why 1?!) is meaningless.
    //the same story with Mikolov's code. It explains why key independent does not change results.

    neu1.ac.setConstant(1.0);

    copyHiddenLayerToInput();

    if (bptt>0)
    {
        for (a=1; a<bptt+bptt_block; a++) bptt_history_v[a]=0;
        for (a=1; a<bptt+bptt_block; a++) bptt_history_m[a]=0;
        for (a=bptt+bptt_block-1; a>1; a--)
        {
            bptt_hidden[a].ac.setZero();
            bptt_hidden[a].ac.setZero();
        }
    }
}

void RnnlmRussianMorphology::computeRecurrentLayer_(int i_word, int i_morph)
{
    if(i_word != -1)
    {
        neu1.ac = syn0h * neu0.ac;
        neu1.ac.addMatrixColumn(syn0v, i_word);
        neu1.ac.addMatrixColumn(syn0m, i_morph);
    }
    else
    {
        neu1.ac = syn0h * neu0.ac;
    }
    neu1.ac.logisticActivation();
}

void RnnlmRussianMorphology::computeOutputLayer_()
{
    neu2v.ac = syn1v * neu1.ac;
    neu2v.ac.softmaxActivation();
    neu2m.ac = syn1m * neu1.ac;
    neu2m.ac.softmaxActivation();
    //neu2v.ac.print();
   // neu2m.ac.print();
}

void RnnlmRussianMorphology::computeErrorOnOutput_(int i_trueWord, int i_trueMorph)
{
    neu2v.fastOutputError(i_trueWord, 1-tau);
    neu2m.fastOutputError(i_trueMorph, tau);
}

void RnnlmRussianMorphology::computeErrorOnPrevious_(const Layer& i_nextLayer, Matrix& i_synMat, Layer& i_prevLayer)
{
    i_prevLayer.er = i_synMat.transpose() * i_nextLayer.er;
    i_synMat.transpose();
}

void RnnlmRussianMorphology::computeErrorOnHidden_(const Layer& i_outVLayer, const Layer& i_outMLayer, Matrix& i_synVMat, Matrix& i_synMMat, Layer& i_hiddenLayer)
{
    i_hiddenLayer.er = i_synVMat.transpose() * i_outVLayer.er;
    i_synVMat.transpose();
    i_hiddenLayer.er += i_synMMat.transpose() * i_outMLayer.er;
    i_synMMat.transpose();
}

void RnnlmRussianMorphology::applyGradient_(double i_lr, const Vector& i_next, const Vector& i_prev, Matrix& i_mat, double beta)
{
    i_mat.fastGradUpdate(i_lr, i_next, i_prev, beta);
}

void RnnlmRussianMorphology::applyGradient_(double i_lr, const Vector& i_next, int i_column, Matrix& o_mat, double beta)
{
    o_mat.fastGradColumnUpdate(i_lr, i_next, i_column, beta);
}

void RnnlmRussianMorphology::addGradient_(const Matrix& i_update, Matrix& o_mat , double beta2)
{
    o_mat.addExpression(i_update, beta2);
}

void RnnlmRussianMorphology::applyGradient_(const Matrix& i_update, int i_updateColumn, Matrix& io_target, int i_targetColumn, double i_beta)
{
    io_target.addColumnToColumn(i_targetColumn, i_update, i_updateColumn, i_beta);
}

void RnnlmRussianMorphology::incremetLogProbByWordLP_(int word, int morph)
{
    m_logProb += neu2v.ac.elementLog(word)*(1-tau);
    m_logProb += neu2m.ac.elementLog(morph)*tau;
}

void RnnlmRussianMorphology::updateBptt_(int last_word, int last_morph) //shift memory needed for bptt to next time step
{
    for (int a=bptt+bptt_block-1; a>0; a--)
    {
        bptt_history_v[a]=bptt_history_v[a-1];
        bptt_history_m[a]=bptt_history_m[a-1];
    }
    bptt_history_v[0]=last_word;
    bptt_history_m[0]=last_morph;

    Vector tmp_ac = std::move(bptt_hidden[bptt+bptt_block-1].ac);
    Vector tmp_er = std::move(bptt_hidden[bptt+bptt_block-1].er);

    for (int a=bptt+bptt_block-1; a>0; a--)
    {
          bptt_hidden[a].ac = std::move(bptt_hidden[a-1].ac);
          bptt_hidden[a].er = std::move(bptt_hidden[a-1].er);
    }
    bptt_hidden[0].ac = std::move(tmp_ac);
    bptt_hidden[0].er = std::move(tmp_er);
}

void RnnlmRussianMorphology::makeBptt_(int word, int morph, double alpha, double beta, int counter)
{
    double beta2, beta3;

    beta2=beta*alpha;
    beta3=beta2*1;

    bptt_hidden[0].ac=neu1.ac;
    bptt_hidden[0].er=neu1.er;

    if (!((counter%bptt_block==0) || (m_independent && (word==0))))
        return;

    for (int step=0; step < bptt+bptt_block-2; step++)
    {
        neu1.logisticErrorActivation();
        //weight update 1->0
        int w = bptt_history_v[step];
        int m = bptt_history_m[step];
        if (w!=-1)
        {
            applyGradient_(alpha, neu1.er, w, bptt_syn0v, 0);
            applyGradient_(alpha, neu1.er, 1, bptt_syn0m, 0);
        }

        computeErrorOnPrevious_(neu1,syn0h,neu0);
        applyGradient_(alpha, neu1.er, neu0.ac, bptt_syn0h, 0);

        //propagate error from time T-n to T-n-1
        neu1.er = neu0.er + bptt_hidden[step+1].er;

        if (step<bptt+bptt_block-3)
        {
            neu1.ac = bptt_hidden[step+1].ac;
            neu0.ac = bptt_hidden[step+2].ac;
        }
    }

    for (int a = 0; a< bptt + bptt_block; a++)
    {
        bptt_hidden[a].er.setZero();
    }

    neu1.ac = bptt_hidden[0].ac;		//restore hidden layer after bptt

    if ((counter%10)==0)
    {
        addGradient_(bptt_syn0h,syn0h,beta2);
        bptt_syn0h.setZero();
    }
    else
    {
        addGradient_(bptt_syn0h,syn0h,beta2);
        bptt_syn0h.setZero();
    }

    if ((counter%10)==0)
    {
        for (int step=0; step<bptt+bptt_block-2; step++)
            if (bptt_history_v[step]!=-1)
            {
                applyGradient_(bptt_syn0v, bptt_history_v[step], syn0v, bptt_history_v[step], beta2);
                bptt_syn0v.setZeroColumn(bptt_history_v[step]);
                applyGradient_(bptt_syn0m, bptt_history_m[step], syn0m, bptt_history_m[step], beta2);
                bptt_syn0m.setZeroColumn(bptt_history_m[step]);
            }
    }
    else
    {
        for (int step=0; step<bptt+bptt_block-2; step++)
            if (bptt_history_v[step]!=-1)
            {
                applyGradient_(bptt_syn0v, bptt_history_v[step], syn0v, bptt_history_v[step], 0);
                bptt_syn0v.setZeroColumn(bptt_history_v[step]);
                applyGradient_(bptt_syn0m, bptt_history_m[step], syn0m, bptt_history_m[step], 0);
                bptt_syn0m.setZeroColumn(bptt_history_m[step]);
            }
    }

}

void RnnlmRussianMorphology::readFromFile(FILE *fi, FileTypeEnum filetype)
{
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &m_vocabSize);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &m_morphologySize);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer1_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt_block);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &m_independent);
    //
    initNet_();
    //
    double d;
    float fl;
    if (filetype==TEXT) {
    goToDelimiter(':', fi);
    for (int a=0; a<layer1_size; a++) {
        fscanf(fi, "%lf", &d);
        neu1.ac[a]=d;
    }
    }
    if (filetype==BINARY) {
    fgetc(fi);
    for (int a=0; a<layer1_size; a++) {
        fread(&fl, 4, 1, fi);
        neu1.ac[a]=fl;
    }
    }

    neu1.ac.update();

    double* syn0v_init=(double *)calloc(layer1_size*m_vocabSize, sizeof(double));
    double* syn0h_init=(double *)calloc(layer1_size*layer1_size, sizeof(double));
    double* syn0m_init=(double *)calloc(layer1_size*m_morphologySize, sizeof(double));
    double* syn1v_init = (double *)calloc(layer1_size*m_vocabSize, sizeof(double));
    double* syn1m_init = (double *)calloc(layer1_size*m_morphologySize, sizeof(double));

    //
    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_vocabSize; a++)
            {
                fscanf(fi, "%lf", &d);
                syn0v_init[a+b*m_vocabSize]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a < m_vocabSize; a++)
            {
                fread(&fl, 4, 1, fi);
                syn0v_init[a+b*m_vocabSize]=fl;
            }
        }
    }

    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_morphologySize; a++)
            {
                fscanf(fi, "%lf", &d);
                syn0m_init[a+b*m_morphologySize]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a < m_morphologySize; a++)
            {
                fread(&fl, 4, 1, fi);
                syn0m_init[a+b*m_morphologySize]=fl;
            }
        }
    }

    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn0h_init[a+b*layer1_size]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a < layer1_size; a++)
            {
                fread(&fl, 4, 1, fi);
                syn0h_init[a+b*layer1_size]=fl;
            }
        }
    }

    //
    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (int b=0; b<m_vocabSize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn1v_init[a+b*layer1_size]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<m_vocabSize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fread(&fl, 4, 1, fi);
                syn1v_init[a+b*layer1_size]=fl;
            }
        }
    }
    //
    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (int b=0; b<m_morphologySize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn1m_init[a+b*layer1_size]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<m_morphologySize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fread(&fl, 4, 1, fi);
                syn1m_init[a+b*layer1_size]=fl;
            }
        }
    }

    syn0v.setMatrix(syn0v_init, layer1_size, m_vocabSize);
    syn0m.setMatrix(syn0m_init, layer1_size, m_morphologySize);
    syn0h.setMatrix(syn0h_init, layer1_size, layer1_size);
    syn1v.setMatrix(syn1v_init, m_vocabSize, layer1_size);
    syn1m.setMatrix(syn1m_init, m_morphologySize, layer1_size);

//    neu1.ac.print();
//    syn0v.print();
//    syn0m.print();
//    syn0h.print();
//    syn1v.print();
//    syn1m.print();

    free(syn0v_init);
    free(syn0m_init);
    free(syn0h_init);
    free(syn1v_init);
    free(syn1m_init);

    saveWeights();
}

void RnnlmRussianMorphology::writeToFile(FILE *fo, FileTypeEnum filetype)
{
//    neu1.ac.print();
//    syn0v.print();
//    syn0m.print();
//    syn0h.print();
//    syn1v.print();
//    syn1m.print();

    float fl = 0;
    fprintf(fo, "Model\n");
    fprintf(fo, "vocabulary size: %d\n", m_vocabSize);
    fprintf(fo, "morphology size: %d\n", m_morphologySize);
    fprintf(fo, "hidden layer size: %d\n", layer1_size);

    fprintf(fo, "bptt: %d\n", bptt);
    fprintf(fo, "bptt block: %d\n", bptt_block);
    fprintf(fo, "independent sentences mode: %d\n", m_independent);
    neu1.ac.prepareToSave();
    if (filetype==TEXT)
    {
        fprintf(fo, "\nHidden layer activation:\n");
        for (int a=0; a<layer1_size; a++) fprintf(fo, "%.4f\n", neu1.ac[a]);
    }
    if (filetype==BINARY)
    {
        for (int a=0; a<layer1_size; a++)
        {
            fl=neu1.ac[a];
            fwrite(&fl, 4, 1, fo);
        }
    }
    //////////
    syn0v.prepareToSave();
    syn0m.prepareToSave();
    syn0h.prepareToSave();
    syn1v.prepareToSave();
    syn1m.prepareToSave();

//    syn0v.print();
//    syn0h.print();
//    syn1.print();
//    neu1.ac.print();

    if (filetype==TEXT)
    {
        fprintf(fo, "\nVocab.Weights 0->1:\n");
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_vocabSize; a++)
            {
                fprintf(fo, "%.4f\n", syn0v.getElement(b,a));
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_vocabSize; a++)
            {
                fl=syn0v.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    ////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\Morph.Weights 0->1:\n");
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_morphologySize; a++)
            {
                fprintf(fo, "%.4f\n", syn0m.getElement(b,a));
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<m_morphologySize; a++)
            {
                fl=syn0m.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    ////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\nRec.Weights 0->1:\n");
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fprintf(fo, "%.4f\n", syn0h.getElement(b,a));
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<layer1_size; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fl=syn0h.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    /////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\n\nVocab.Weights 1->2:\n");
        for (int b=0; b<m_vocabSize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fprintf(fo, "%.4f\n", syn1v.getElement(b,a));
            }
        }

    }
    if (filetype==BINARY)
    {
        for (int b=0; b<m_vocabSize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fl=syn1v.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    ////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\n\nMorph.Weights 1->2:\n");
        for (int b=0; b<m_morphologySize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fprintf(fo, "%.4f\n", syn1m.getElement(b,a));
            }
        }
    }
    if (filetype==BINARY)
    {
        for (int b=0; b<m_morphologySize; b++)
        {
            for (int a=0; a<layer1_size; a++)
            {
                fl=syn1m.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    /////////
}

}
