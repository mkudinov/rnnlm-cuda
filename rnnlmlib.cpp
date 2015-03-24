///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"
#include <sys/time.h>
#include <iostream>

#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

double CRnnLM::random(double min, double max)
{
    return rand()/(double)RAND_MAX*(max-min)+min;
}

void CRnnLM::setTrainFile(char *str)
{
    strcpy(train_file, str);
}

void CRnnLM::setValidFile(char *str)
{
    strcpy(valid_file, str);
}

void CRnnLM::setTestFile(char *str)
{
    strcpy(test_file, str);
}

void CRnnLM::setRnnLMFile(char *str)
{
    strcpy(rnnlm_file, str);
}


void CRnnLM::readWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) {
    ch=fgetc(fin);

    if (ch==13) continue;

    if ((ch==' ') || (ch=='\t') || (ch=='\n')) {
            if (a>0) {
                if (ch=='\n') ungetc(ch, fin);
                break;
            }

            if (ch=='\n') {
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


/* Version for bad corpora
void CRnnLM::readWord(char *word, FILE *fin) {
    int a = 0, ch, nextch;
    while (!feof(fin)) {
      ch=fgetc(fin);

      if (ch==13) continue;

      if ((ch=='-') || (ch==',') || (ch=='.')) {
        if (a==0) continue;
        nextch=fgetc(fin);
        if ((nextch==' ') || (nextch=='\t') || (nextch=='\n') || (nextch=='-') || (ch=='.') ) {
          ungetc(nextch, fin);
          break;
        }
        ungetc(nextch, fin);
      }

      if ((ch==' ') || (ch=='\t') || (ch=='\n') || (ch=='[') || (ch==']') || (ch=='!') || (ch=='"') ||
        (ch=='…') || (ch=='–') || (ch=='—') || (ch=='?') || (ch=='(') || (ch==')') || (ch==';') || (ch==':') || (ch=='»') || (ch=='«')) {
        if (a>0) {
          break;
        }
        else
          continue;
        }

        word[a]=ch;
        a++;

        if (a>=MAX_STRING) {
              //printf("Too long word found!\n");   //truncate too long words
          a--;
        }
      }
    word[a] = 0;
}
*/

int CRnnLM::getWordHash(char *word)
{
    unsigned int hash, a;

    hash=0;
    for (a=0; a<strlen(word); a++) hash=hash*237+word[a];
    hash=hash%vocab_hash_size;

    return hash;
}

int CRnnLM::searchVocab(char *word)
{
    int a;
    unsigned int hash;

    hash=getWordHash(word);

    if (vocab_hash[hash]==-1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];

    for (a=0; a<vocab_size; a++) {				//search in vocabulary
        if (!strcmp(word, vocab[a].word)) {
            vocab_hash[hash]=a;
            return a;
        }
    }

    return -1;							//return OOV if not found
}

int CRnnLM::readWordIndex(FILE *fin)
{
    char word[MAX_STRING];

    readWord(word, fin);
    if (feof(fin)) return -1;

    return searchVocab(word);
}

int CRnnLM::addWordToVocab(char *word)
{
    unsigned int hash;

    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn=0;
    vocab_size++;

    if (vocab_size+2>=vocab_max_size) {        //reallocate memory if needed
        vocab_max_size+=100;
        vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }

    hash=getWordHash(word);
    vocab_hash[hash]=vocab_size-1;

    return vocab_size-1;
}

void CRnnLM::sortVocab()
{
    int a, b, max;
    vocab_word swap;

    for (a=1; a<vocab_size; a++) {
        max=a;
        for (b=a+1; b<vocab_size; b++) if (vocab[max].cn<vocab[b].cn) max=b;

        swap=vocab[max];
        vocab[max]=vocab[a];
        vocab[a]=swap;
    }
}

void CRnnLM::learnVocabFromTrainFile()    //assumes that vocabulary is empty
{
    char word[MAX_STRING];
    FILE *fin;
    int a, i, train_wcn;

    for (a=0; a<vocab_hash_size; a++) vocab_hash[a]=-1;

    fin=fopen(train_file, "rb");

    vocab_size=0;

    addWordToVocab((char *)"</s>");

    train_wcn=0;
    while (1) {
        readWord(word, fin);
        if (feof(fin)) break;

        train_wcn++;

        i=searchVocab(word);
        if (i==-1) {
            a=addWordToVocab(word);
            vocab[a].cn=1;
        } else vocab[i].cn++;
    }

    sortVocab();

    //select vocabulary size
    /*a=0;
    while (a<vocab_size) {
    a++;
    if (vocab[a].cn==0) break;
    }
    vocab_size=a;*/

    if (debug_mode>0) {
    printf("Vocab size: %d\n", vocab_size);
    printf("Words in train file: %d\n", train_wcn);
    }

    train_words=train_wcn;

    fclose(fin);
}

void CRnnLM::PrintVocabIntoFile(char* str) {
    char print_file[MAX_STRING];
    strcpy(print_file, str);
    FILE* fw = fopen(print_file, "w");
    fprintf (fw, "%d\n", vocab_size);
    for (int a = 0; a < vocab_size; ++a) {
        fprintf(fw, "%s %d\n", vocab[a].word, vocab[a].cn);
    }
    fclose(fw);
}

void CRnnLM::saveWeights()      //saves current weights and unit activations
{
    neu0b = neu0;
    neu1b = neu1;
    neu2b = neu2;
    syn0vb = syn0v;
    syn0hb = syn0h;
    syn1b = syn1;
}

void CRnnLM::restoreWeights()      //restores current weights and unit activations from backup copy
{
    neu0 = neu0b;
    neu1 = neu1b;

    neu2 = neu2b;

    syn0v = syn0vb;
    syn0h = syn0hb;

    syn1 = syn1b;

}

void CRnnLM::saveContext()		//useful for n-best list processing
{
    //int a;
    //for (a=0; a<layer1_size; a++) neu1b_ac[a]=neu1_ac[a];
    neu1b.ac = neu1.ac;
}

void CRnnLM::restoreContext()
{
//    int a;
//    for (a=0; a<layer1_size; a++) neu1_ac[a]=neu1b_ac[a];
    neu1.ac = neu1b.ac;
}

void CRnnLM::saveContext2()
{
//    int a;

//    for (a=0; a<layer1_size; a++) neu1b2_ac[a]=neu1_ac[a];
    neu1b2.ac = neu1.ac;
}

void CRnnLM::restoreContext2()
{
//    int a;

//    for (a=0; a<layer1_size; a++) neu1_ac[a]=neu1b2_ac[a];
    neu1.ac = neu1b2.ac;
}

void CRnnLM::initNet()
{
    int a, b;

    neu0.setConstant(layer1_size, 0);
    neu1.setConstant(layer1_size, 0);
    neu2.setConstant(vocab_size, 0);

    neu0b.setConstant(layer1_size, 0);
    neu1b.setConstant(layer1_size, 0);
    neu1b2.setConstant(layer1_size, 0);
    neu2b.setConstant(vocab_size, 0);

    syn0vb.setZero(vocab_size, layer1_size);
    syn0hb.setZero(layer1_size,layer1_size);

    syn1b.setZero(layer1_size, vocab_size);

    double* syn0_init=(double *)calloc(layer1_size*(vocab_size+layer1_size), sizeof(double));
    double* syn0m_init=(double *)calloc(layer1_size*(morph_size), sizeof(double));
    double* syn1_init = (double *)calloc(layer1_size*(vocab_size+class_size), sizeof(double));
    double* syn0v_init=(double *)calloc(layer1_size*vocab_size, sizeof(double));
    double* syn0h_init=(double *)calloc(layer1_size*layer1_size, sizeof(double));

    for (b=0; b<layer1_size; b++) for (a=0; a<vocab_size+layer1_size; a++)
    {
        syn0_init[a+b*(vocab_size+layer1_size)]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); //init block intended to initialize the same matrices as in mikolov's implementation (for debug purposes)
    }

    for (b=0; b<vocab_size+1; b++) for (a=0; a<layer1_size; a++)
    {
        syn1_init[a+b*layer1_size]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<morph_size; a++)
    {
        syn0_init[a+b*(morph_size)]=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1); //init block intended to initialize the same matrices as in mikolov's implementation (for debug purposes)
    }

    for(int i = 0; i < layer1_size; i++)
    {
        for(int j = 0; j < vocab_size; j++)
        {
            syn0v_init[i * vocab_size + j] = syn0_init[j + (vocab_size+layer1_size) * i];
        }
    }

    for(int i = 0; i < layer1_size; i++)
    {
        for(int j = 0; j < layer1_size; j++)
        {
            syn0h_init[i * layer1_size + j] = syn0_init[vocab_size + j + (vocab_size+layer1_size) * i];
        }
    }

    free(syn0_init);

    syn0v.setMatrix(syn0v_init, layer1_size, vocab_size);
    syn0h.setMatrix(syn0h_init, layer1_size, layer1_size);
    syn0m.setMatrix(syn0m_init, layer1_size, morph_size);
    syn1.setMatrix(syn1_init, vocab_size, layer1_size);

    free(syn0v_init);
    free(syn0h_init);
    free(syn0m_init);
    free(syn1_init);

    if (bptt>0)
    {
        bptt_history = std::vector<int>(bptt+bptt_block+10, -1);

        for (size_t i=0; i < bptt + bptt_block; i++)
        {
            Layer newLayer;
            newLayer.setConstant(layer1_size, 0);
            bptt_hidden.push_back(std::move(newLayer));
        }

        bptt_syn0v.setZero(layer1_size, vocab_size);
        bptt_syn0m.setZero(layer1_size, vocab_size);
        bptt_syn0h.setZero(layer1_size, layer1_size);
    }

    saveWeights();
}

void CRnnLM::saveNet()       //will save the whole network structure
{
    FILE *fo;
    int a, b;
    char str[1000];
    float fl;

    sprintf(str, "%s.temp", rnnlm_file);

    fo=fopen(str, "wb");
    if (fo==NULL) {
        printf("Cannot create file %s\n", rnnlm_file);
        exit(1);
    }
    fprintf(fo, "version: %d\n", version);
    fprintf(fo, "file format: %d\n\n", filetype);

    fprintf(fo, "training data file: %s\n", train_file);
    fprintf(fo, "validation data file: %s\n\n", valid_file);

    fprintf(fo, "last probability of validation data: %f\n", llogp);
    fprintf(fo, "number of finished iterations: %d\n", iter);

    fprintf(fo, "current position in training data: %d\n", train_cur_pos);
    fprintf(fo, "current probability of training data: %f\n", logp);
    fprintf(fo, "save after processing # words: %d\n", anti_k);
    fprintf(fo, "# of training words: %d\n", train_words);

    fprintf(fo, "input layer size: %d\n", vocab_size+layer1_size);
    fprintf(fo, "hidden layer size: %d\n", layer1_size);
    fprintf(fo, "compression layer size: %d\n", 0);
    fprintf(fo, "output layer size: %d\n", vocab_size + 1);

    fprintf(fo, "direct connections: %lld\n", direct_size);
    fprintf(fo, "direct order: %d\n", direct_order);

    fprintf(fo, "bptt: %d\n", bptt);
    fprintf(fo, "bptt block: %d\n", bptt_block);

    fprintf(fo, "vocabulary size: %d\n", vocab_size);
    fprintf(fo, "class size: %d\n", class_size);

    fprintf(fo, "old classes: %d\n", old_classes);
    fprintf(fo, "independent sentences mode: %d\n", independent);

    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");

    fprintf(fo, "\nVocabulary:\n");
    for (a=0; a<vocab_size; a++) fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, vocab[a].cn, vocab[a].word, vocab[a].class_index);

    neu1.ac.prepareToSave();
    if (filetype==TEXT)
    {
        fprintf(fo, "\nHidden layer activation:\n");
        for (a=0; a<layer1_size; a++) fprintf(fo, "%.4f\n", neu1.ac[a]);
    }
    if (filetype==BINARY)
    {
        for (a=0; a<layer1_size; a++)
        {
            fl=neu1.ac[a];
            fwrite(&fl, 4, 1, fo);
        }
    }
    //////////
    syn0v.prepareToSave();
    syn0h.prepareToSave();
    syn1.prepareToSave();

//    syn0v.print();
//    syn0h.print();
//    syn1.print();
//    neu1.ac.print();

    if (filetype==TEXT)
    {
        fprintf(fo, "\nWeights 0->1:\n");
        for (b=0; b<layer1_size; b++)
        {
            for (a=0; a<vocab_size; a++)
            {
                fprintf(fo, "%.4f\n", syn0v.getElement(b,a));
            }
            for (a=0; a<layer1_size; a++)
            {
                fprintf(fo, "%.4f\n", syn0h.getElement(b,a));
            }
        }
    }
    if (filetype==BINARY)
    {
        for (b=0; b<layer1_size; b++)
        {
            for (a=0; a<vocab_size; a++)
            {
                fl=syn0v.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
            for (a=0; a<layer1_size; a++)
            {
                fl=syn0h.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
    }
    /////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\n\nWeights 1->2:\n");
        for (b=0; b<vocab_size; b++)
        {
            for (a=0; a<layer1_size; a++)
            {
                fprintf(fo, "%.4f\n", syn1.getElement(b,a));
            }
        }
        for (a=0; a<layer1_size; a++)
        {
            fprintf(fo, "%.4f\n", 1.0); //we removed classes so for compatibility with the original tool we leave here class weights
        }

    }
    if (filetype==BINARY)
    {
        for (b=0; b<vocab_size; b++)
        {
            for (a=0; a<layer1_size; a++)
            {
                fl=syn1.getElement(b,a);
                fwrite(&fl, 4, 1, fo);
            }
        }
        for (a=0; a<layer1_size; a++)
        {
            fl=1.0;
            fwrite(&fl, 4, 1, fo); //same here
        }
    }
    ////////
    if (filetype==TEXT)
    {
        fprintf(fo, "\nDirect connections:\n"); //for compatibility
    }
    ////////
    fclose(fo);

    rename(str, rnnlm_file);
}

void CRnnLM::goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(fi);
        if (feof(fi)) {
            printf("Unexpected end of file\n");
            exit(1);
        }
    }
}

void CRnnLM::restoreNet()    //will read whole network structure
{
    FILE *fi;
    int a, b, ver;
    float fl;
    char str[MAX_STRING];
    double d;

    int layer0_size_buff,layerc_size_buff,layer2_size_buff;

    fi=fopen(rnnlm_file, "rb");
    if (fi==NULL) {
    printf("ERROR: model file '%s' not found!\n", rnnlm_file);
    exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (version==5)) /* we will solve this later.. */ ; else
    if (ver!=version) {
        printf("Unknown version of file %s\n", rnnlm_file);
        exit(1);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    goToDelimiter(':', fi);
    if (train_file_set==0) {
    fscanf(fi, "%s", train_file);
    } else fscanf(fi, "%s", str);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", valid_file);
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
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &anti_k);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer0_size_buff);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer1_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layerc_size_buff);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer2_size_buff);
    //
    if (ver>5) {
    goToDelimiter(':', fi);
    fscanf(fi, "%lld", &direct_size);
    }
    //
    if (ver>6) {
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &direct_order);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt);
    //
    if (ver>4) {
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt_block);
    } else bptt_block=10;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &vocab_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &class_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &old_classes);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &independent);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    starting_alpha=d;
    //
    goToDelimiter(':', fi);
    if (alpha_set==0) {
    fscanf(fi, "%lf", &d);
    alpha=d;
    } else fscanf(fi, "%lf", &d);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
    //


    //read normal vocabulary
    if (vocab_max_size<vocab_size) {
    if (vocab!=NULL) free(vocab);
        vocab_max_size=vocab_size+1000;
        vocab=(struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));    //initialize memory for vocabulary
    }
    //
    goToDelimiter(':', fi);
    for (a=0; a<vocab_size; a++) {
    //fscanf(fi, "%d%d%s%d", &b, &vocab[a].cn, vocab[a].word, &vocab[a].class_index);
    fscanf(fi, "%d%d", &b, &vocab[a].cn);
    readWord(vocab[a].word, fi);
    fscanf(fi, "%d", &vocab[a].class_index);
    //printf("%d  %d  %s  %d\n", b, vocab[a].cn, vocab[a].word, vocab[a].class_index);
    }
    //
    initNet();
    //

    if (filetype==TEXT) {
    goToDelimiter(':', fi);
    for (a=0; a<layer1_size; a++) {
        fscanf(fi, "%lf", &d);
        neu1.ac[a]=d;
    }
    }
    if (filetype==BINARY) {
    fgetc(fi);
    for (a=0; a<layer1_size; a++) {
        fread(&fl, 4, 1, fi);
        neu1.ac[a]=fl;
    }
    }

    neu1.ac.update();

    double* syn0v_init=(double *)calloc(layer1_size*vocab_size, sizeof(double));
    double* syn0h_init=(double *)calloc(layer1_size*layer1_size, sizeof(double));
    double* syn1_init = (double *)calloc(layer1_size*(vocab_size+class_size), sizeof(double));

    //
    if (filetype==TEXT)
    {
        goToDelimiter(':', fi);
        for (b=0; b<layer1_size; b++)
        {
            for (a=0; a<vocab_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn0v_init[a+b*vocab_size]=d;
            }
            for (a=0; a<layer1_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn0h_init[a+b*layer1_size]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (b=0; b<layer1_size; b++)
        {
            for (a=0; a<vocab_size; a++)
            {
                fread(&fl, 4, 1, fi);
                syn0v_init[a+b*vocab_size]=fl;
            }
            for (a=0; a<layer1_size; a++)
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
        for (b=0; b<vocab_size + class_size; b++)
        {
            for (a=0; a<layer1_size; a++)
            {
                fscanf(fi, "%lf", &d);
                syn1_init[a+b*layer1_size]=d;
            }
        }
    }
    if (filetype==BINARY)
    {
        for (b=0; b<layer1_size + 1; b++)
        {
            for (a=0; a<layer1_size; a++)
            {
                fread(&fl, 4, 1, fi);
                syn1_init[a+b*layer1_size]=fl;
            }
        }
    }

    syn0v.setMatrix(syn0v_init, layer1_size, vocab_size);
    syn0h.setMatrix(syn0h_init, layer1_size, layer1_size);
    syn1.setMatrix(syn1_init, vocab_size, layer1_size);

    free(syn0v_init);
    free(syn0h_init);
    free(syn1_init);

    saveWeights();

    fclose(fi);
}

void CRnnLM::netFlush()   //cleans all activations and error vectors
{
    neu0.ac.setConstant(0.1);
    neu0.er.setZero();

    neu1.ac.setZero();
    neu1.er.setZero();

    neu2.ac.setZero();
    neu2.er.setZero();
}

void CRnnLM::netReset()   //cleans hidden layer activation + bptt history
{
    int a, b;

    neu1.ac.setConstant(1.0);

    copyHiddenLayerToInput();

    if (bptt>0)
    {
        for (a=1; a<bptt+bptt_block; a++) bptt_history[a]=0;
        for (a=bptt+bptt_block-1; a>1; a--)
        {
            bptt_hidden[a].ac.setZero();
            bptt_hidden[a].ac.setZero();
        }
    }

    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
}

void CRnnLM::computeNet(int last_word, int word)
{
    computeRecurrentLayer_(last_word);
    if (word!=-1)
    {
        computeOutputLayer_();
    }
}

void CRnnLM::learnNet(int last_word, int word)
{
    double beta2, beta3;

    beta2=beta*alpha;
    beta3=beta2*1;	//beta3 can be possibly larger than beta2, as that is useful on small datasets (if the final model is to be interpolated wich backoff model) - todo in the future

    if (word==-1) return;

    computeErrorOnOutput_(word);

    computeErrorOnPrevious_(neu2, syn1, neu1);

    if ((counter%10)==0)
        applyGradient_(alpha, neu2.er, neu1.ac, syn1, beta2);
    else
        applyGradient_(alpha, neu2.er, neu1.ac, syn1, 0);

  //  neu1.ac.print();

  //  neu2.er.print();

  //  syn1.print();

    if (bptt<=1)
    {
        neu1.softmaxErrorActivation();

        //weight update 1->0
        if (last_word!=-1)
        {
            if ((counter%10)==0)
                applyGradient_(alpha, neu1.er, last_word, syn0v, beta2);
            else
                applyGradient_(alpha, neu1.er, last_word, syn0v, 0);
        }

        if ((counter%10)==0)
            applyGradient_(alpha,neu1.er,neu0.ac,syn0h,beta2);
        else
            applyGradient_(alpha,neu1.er,neu0.ac,syn0h,0);
    }
    else		//BPTT
    {
        bptt_hidden[0].ac=neu1.ac;
        bptt_hidden[0].er=neu1.er;

        if (((counter%bptt_block)==0) || (independent && (word==0)))
        {
            for (int step=0; step < bptt+bptt_block-2; step++)
            {
                neu1.softmaxErrorActivation();
                //weight update 1->0
                int a = bptt_history[step];
                if (a!=-1)
                {
                    applyGradient_(alpha,neu1.er,a,bptt_syn0v,0);
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
                    if (bptt_history[step]!=-1)
                    {
                        applyGradient_(bptt_syn0v, bptt_history[step], syn0v, bptt_history[step], beta2);
                        bptt_syn0v.setZeroColumn(bptt_history[step]);
                    }
            }
            else
            {
                for (int step=0; step<bptt+bptt_block-2; step++) if (bptt_history[step]!=-1)
                {
                    applyGradient_(bptt_syn0v, bptt_history[step], syn0v, bptt_history[step], 0);
                    bptt_syn0v.setZeroColumn(bptt_history[step]);
                }
            }

 //           syn0h.print();

        }
    }
}

void CRnnLM::copyHiddenLayerToInput()
{
    neu0.ac = neu1.ac;
}

void CRnnLM::trainNet()
{
    int word, last_word, wordcn;
    char log_name[200];
    FILE *fi, *flog;
    clock_t start, now;

    sprintf(log_name, "%s.output.txt", rnnlm_file);

    printf("Starting training using file %s\n", train_file);
    starting_alpha=alpha;

    fi=fopen(rnnlm_file, "rb");
    if (fi!=NULL)
    {
        fclose(fi);
        printf("Restoring network from file to continue training...\n");
        restoreNet();
    }
    else
    {
        learnVocabFromTrainFile();
        initNet();
        iter=0;
    }

    if (class_size>vocab_size)
    {
        printf("WARNING: number of classes exceeds vocabulary size!\n");
    }

    counter=train_cur_pos;

    //saveNet();

    while (1)
    {
        printf("Iter: %3d\tAlpha: %f\t   ", iter, alpha);
        fflush(stdout);

        if (bptt>0) for (int a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
        for (int a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

        //TRAINING PHASE
        netFlush();

        fi=fopen(train_file, "rb");
        last_word=0;

        if (counter>0) for (int a=0; a<counter; a++) word=readWordIndex(fi);	//this will skip words that were already learned if the training was interrupted

        start=clock();

        while (1) {
            counter++;

            if ((counter%10000)==0) if ((debug_mode>1)) {
            now=clock();
            if (train_words>0)
                printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %.2f%%   Words/sec: %.1f ", 13, iter, alpha, -logp/log10(2)/counter, counter/(double)train_words*100, counter/((double)(now-start)/1000000.0));
            else
                printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Progress: %dK", 13, iter, alpha, -logp/log10(2)/counter, counter/1000);
            fflush(stdout);
            }

            if ((anti_k>0) && ((counter%anti_k)==0)) {
            train_cur_pos=counter;
            saveNet();
            }

        word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: test on validation data, iterate till convergence

//            if (word!=-1) logp+=log10(neu2.ac[word]);

//            if ((logp!=logp) || (isinf(logp))) {
//                printf("\nNumerical error %d %f\n", word, neu2.ac[word]);
//                exit(1);
//            }


            if (bptt>0)
            {		//shift memory needed for bptt to next time step
                for (int a=bptt+bptt_block-1; a>0; a--)
                {
                    bptt_history[a]=bptt_history[a-1];
                }
                bptt_history[0]=last_word;

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

            learnNet(last_word, word);

            copyHiddenLayerToInput();

         //   if (last_word!=-1) neu0.ac[last_word]=0;  //delete previous activation

            last_word=word;

            for (int a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
            history[0]=last_word;

        if (independent && (word==0)) netReset();
        }
        fclose(fi);

    now=clock();
        printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -logp/log10(2)/counter, counter/((double)(now-start)/1000000.0));

        if (one_iter==1) {	//no validation data are needed and network is always saved with modified weights
            printf("\n");
        logp=0;
            saveNet();
            break;
        }

        //VALIDATION PHASE
        netFlush();

        fi=fopen(valid_file, "rb");
    if (fi==NULL) {
        printf("Valid file not found\n");
        exit(1);
    }

        flog=fopen(log_name, "ab");
    if (flog==NULL) {
        printf("Cannot open log file\n");
        exit(1);
    }

        //fprintf(flog, "Index   P(NET)          Word\n");
        //fprintf(flog, "----------------------------------\n");

        last_word=0;
        logp=0;
        wordcn=0;
        while (1) {
            word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: report LOGP, PPL

            if (word!=-1) {
            logp+=log10(neu2.ac[word]);
            wordcn++;
            }

            /*if (word!=-1)
                fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
            else
                fprintf(flog, "-1\t0\t\tOOV\n");*/

            //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
            copyHiddenLayerToInput();

            last_word=word;

            for (int a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
            history[0]=last_word;

        if (independent && (word==0)) netReset();

        }
        fclose(fi);

        fprintf(flog, "\niter: %d\n", iter);
        fprintf(flog, "valid log probability: %f\n", logp);
        fprintf(flog, "PPL net: %f\n", exp10(-logp/(double)wordcn));

        fclose(flog);

        printf("VALID entropy: %.4f\n", -logp/log10(2)/wordcn);

        counter=0;
    train_cur_pos=0;

        if (logp<llogp)
            restoreWeights();
        else
            saveWeights();

        if (logp*min_improvement<llogp) {
            if (alpha_divide==0)
                alpha_divide=1;
            else {
                saveNet();
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        iter++;
        saveNet();
    }
}

void CRnnLM::testNet()
{
////    int a, b, word, last_word, wordcn;
////    FILE *fi, *flog, *lmprob=NULL;
////    real prob_other, log_other, log_combine;
////    double d;

////    restoreNet();

////    if (use_lmprob) {
////    lmprob=fopen(lmprob_file, "rb");
////    }

////    //TEST PHASE
////    //netFlush();

////    fi=fopen(test_file, "rb");
////    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
////    //flog=fopen(str, "wb");
////    flog=stdout;

////    if (debug_mode>1)	{
////    if (use_lmprob) {
////            fprintf(flog, "Index   P(NET)          P(LM)           Word\n");
////            fprintf(flog, "--------------------------------------------------\n");
////    } else {
////            fprintf(flog, "Index   P(NET)          Word\n");
////            fprintf(flog, "----------------------------------\n");
////    }
////    }

////    last_word=0;					//last word = end of sentence
////    logp=0;
////    log_other=0;
////    log_combine=0;
////    prob_other=0;
////    wordcn=0;
////    copyHiddenLayerToInput();

////    if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
////    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
////    if (independent) netReset();

////    while (1) {

////        word=readWordIndex(fi);		//read next word
////        computeNet(last_word, word);		//compute probability distribution
////        if (feof(fi)) break;		//end of file: report LOGP, PPL

////        if (use_lmprob) {
////            fscanf(lmprob, "%lf", &d);
////            prob_other=d;

////            goToDelimiter('\n', lmprob);
////        }

////        if ((word!=-1) || (prob_other>0)) {
////            if (word==-1) {
////            logp+=-8;		//some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
////            log_combine+=log10(0 * lambda + prob_other*(1-lambda));
////            } else {
////            logp+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]);
////            log_combine+=log10(neu2_ac[vocab[word].class_index+vocab_size] * neu2_ac[word]*lambda + prob_other*(1-lambda));
////            }
////            log_other+=log10(prob_other);
////            wordcn++;
////        }

////    if (debug_mode>1) {
////            if (use_lmprob) {
////            if (word!=-1) fprintf(flog, "%d\t%.10f\t%.10f\t%s", word, neu2_ac[vocab[word].class_index+vocab_size] *neu2_ac[word], prob_other, vocab[word].word);
////            else fprintf(flog, "-1\t0\t\t0\t\tOOV");
////            } else {
////            if (word!=-1) fprintf(flog, "%d\t%.10f\t%s", word, neu2_ac[vocab[word].class_index+vocab_size] *neu2_ac[word], vocab[word].word);
////            else fprintf(flog, "-1\t0\t\tOOV");
////            }

////            fprintf(flog, "\n");
////        }

////        if (dynamic>0) {
////            if (bptt>0) {
////                for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
////                bptt_history[0]=last_word;

////                for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
////                    bptt_hidden_ac[a*layer1_size+b]=bptt_hidden_ac[(a-1)*layer1_size+b];
////                    bptt_hidden_er[a*layer1_size+b]=bptt_hidden_er[(a-1)*layer1_size+b];
////            }
////            }
////            //
////            alpha=dynamic;
////            learnNet(last_word, word);    //dynamic update
////        }
////        copyHiddenLayerToInput();

////        if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

////        last_word=word;

////        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
////        history[0]=last_word;

////    if (independent && (word==0)) netReset();
////    }
////    fclose(fi);
////    if (use_lmprob) fclose(lmprob);

////    //write to log file
////    if (debug_mode>0) {
////    fprintf(flog, "\ntest log probability: %f\n", logp);
////    if (use_lmprob) {
////            fprintf(flog, "test log probability given by other lm: %f\n", log_other);
////            fprintf(flog, "test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
////    }

////    fprintf(flog, "\nPPL net: %f\n", exp10(-logp/(real)wordcn));
////    if (use_lmprob) {
////            fprintf(flog, "PPL other: %f\n", exp10(-log_other/(real)wordcn));
////            fprintf(flog, "PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
////    }
////    }

////    fclose(flog);
}

void CRnnLM::testNbest()
{
//    int a, word, last_word, wordcn;
//    FILE *fi, *flog, *lmprob=NULL;
//    float prob_other; //has to be float so that %f works in fscanf
//    real log_other, log_combine, senp;
//    //int nbest=-1;
//    int nbest_cn=0;
//    char ut1[MAX_STRING], ut2[MAX_STRING];

//    restoreNet();
//    computeNet(0, 0);
//    copyHiddenLayerToInput();
//    saveContext();
//    saveContext2();

//    if (use_lmprob) {
//    lmprob=fopen(lmprob_file, "rb");
//    } else lambda=1;		//!!! for simpler implementation later

//    //TEST PHASE
//    //netFlush();

//    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

//    if (!strcmp(test_file, "-")) fi=stdin; else fi=fopen(test_file, "rb");

//    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
//    //flog=fopen(str, "wb");
//    flog=stdout;

//    last_word=0;		//last word = end of sentence
//    logp=0;
//    log_other=0;
//    prob_other=0;
//    log_combine=0;
//    wordcn=0;
//    senp=0;
//    strcpy(ut1, (char *)"");
//    while (1) {
//    if (last_word==0) {
//        fscanf(fi, "%s", ut2);

//        if (nbest_cn==1) saveContext2();		//save context after processing first sentence in nbest

//        if (strcmp(ut1, ut2)) {
//        strcpy(ut1, ut2);
//        nbest_cn=0;
//        restoreContext2();
//        saveContext();
//        } else restoreContext();

//        nbest_cn++;

//        copyHiddenLayerToInput();
//        }


//    word=readWordIndex(fi);     //read next word
//    if (lambda>0) computeNet(last_word, word);      //compute probability distribution
//        if (feof(fi)) break;        //end of file: report LOGP, PPL


//        if (use_lmprob) {
//            fscanf(lmprob, "%f", &prob_other);
//            goToDelimiter('\n', lmprob);
//        }

//        if (word!=-1)
//        neu2.ac[word]*=neu2.ac[vocab[word].class_index+vocab_size];

//        if (word!=-1) {
//            logp+=log10(neu2.ac[word]);

//            log_other+=log10(prob_other);

//            log_combine+=log10(neu2.ac[word]*lambda + prob_other*(1-lambda));

//            senp+=log10(neu2.ac[word]*lambda + prob_other*(1-lambda));

//            wordcn++;
//        } else {
//            //assign to OOVs some score to correctly rescore nbest lists, reasonable value can be less than 1/|V| or backoff LM score (in case it is trained on more data)
//            //this means that PPL results from nbest list rescoring are not true probabilities anymore (as in open vocabulary LMs)

//            real oov_penalty=-5;	//log penalty

//            if (prob_other!=0) {
//            logp+=log10(prob_other);
//            log_other+=log10(prob_other);
//            log_combine+=log10(prob_other);
//            senp+=log10(prob_other);
//            } else {
//            logp+=oov_penalty;
//            log_other+=oov_penalty;
//            log_combine+=oov_penalty;
//            senp+=oov_penalty;
//            }
//            wordcn++;
//        }

//        //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
//        copyHiddenLayerToInput();

//       // if (last_word!=-1) neu0.ac[last_word]=0;  //delete previous activation

//        if (word==0) {		//write last sentence log probability / likelihood
//            fprintf(flog, "%f\n", senp);
//            senp=0;
//    }

//        last_word=word;

//        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
//        history[0]=last_word;

//    if (independent && (word==0)) netReset();
//    }
//    fclose(fi);
//    if (use_lmprob) fclose(lmprob);

//    if (debug_mode>0) {
//    printf("\ntest log probability: %f\n", logp);
//    if (use_lmprob) {
//            printf("test log probability given by other lm: %f\n", log_other);
//            printf("test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
//    }

//    printf("\nPPL net: %f\n", exp10(-logp/(real)wordcn));
//    if (use_lmprob) {
//            printf("PPL other: %f\n", exp10(-log_other/(real)wordcn));
//            printf("PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
//    }
//    }

//    fclose(flog);
}

void CRnnLM::testGen()
{
//    std::cout << "The function is turned off" << std::endl;
////    int i, word, cla, last_word, wordcn, c, b, a=0;
////    real f, g, sum, val;

////    restoreNet();

////    word=0;
////    last_word=0;					//last word = end of sentence
////    wordcn=0;
////    copyHiddenLayerToInput();
////    while (wordcn<gen) {
////        computeNet(last_word, 0);		//compute probability distribution

////        f=random(0, 1);
////        g=0;
////        i=vocab_size;
////        while ((g<f) && (i<layer2_size)) {
////            g+=neu2_ac[i];
////            i++;
////        }
////        cla=i-1-vocab_size;

////        if (cla>class_size-1) cla=class_size-1;
////        if (cla<0) cla=0;

////        //
////        // !!!!!!!!  THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCAB !!! (like class 10 = words 11 12 13; not 11 12 16)  !!!!!!!!
////        // forward pass 1->2 for words
////        for (c=0; c<class_cn[cla]; c++) neu2_ac[class_words[cla][c]]=0;
////        matrixXvector(neu2_ac, neu1_ac, syn1, layer1_size, class_words[cla][0], class_words[cla][0]+class_cn[cla], 0, layer1_size, 0);

////        //apply direct connections to words
////    if (word!=-1) if (direct_size>0) {
////            unsigned long long hash[MAX_NGRAM_ORDER];

////            for (a=0; a<direct_order; a++) hash[a]=0;

////            for (a=0; a<direct_order; a++) {
////                b=0;
////                if (a>0) if (history[a-1]==-1) break;
////                hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(cla+1);

////                for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
////                hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
////            }

////            for (c=0; c<class_cn[cla]; c++) {
////            a=class_words[cla][c];

////            for (b=0; b<direct_order; b++) if (hash[b]) {
////                neu2_ac[a]+=syn_d[hash[b]];
////                    hash[b]++;
////                hash[b]=hash[b]%direct_size;
////                } else break;
////            }
////    }

////        //activation 2   --softmax on words
////    sum=0;
////        for (c=0; c<class_cn[cla]; c++) {
////            a=class_words[cla][c];
////            if (neu2_ac[a]>50) neu2_ac[a]=50;  //for numerical stability
////            if (neu2_ac[a]<-50) neu2_ac[a]=-50;  //for numerical stability
////            val=FAST_EXP(neu2_ac[a]);
////            sum+=val;
////            neu2_ac[a]=val;
////        }
////        for (c=0; c<class_cn[cla]; c++) neu2_ac[class_words[cla][c]]/=sum;
////    //

////    f=random(0, 1);
////        g=0;
////        /*i=0;
////        while ((g<f) && (i<vocab_size)) {
////            g+=neu2[i].ac;
////            i++;
////        }*/
////        for (c=0; c<class_cn[cla]; c++) {
////            a=class_words[cla][c];
////            g+=neu2_ac[a];
////            if (g>f) break;
////        }
////        word=a;

////    if (word>vocab_size-1) word=vocab_size-1;
////        if (word<0) word=0;

////    //printf("%s %d %d\n", vocab[word].word, cla, word);
////    if (word!=0)
////        printf("%s ", vocab[word].word);
////    else
////        printf("\n");

////        copyHiddenLayerToInput();

////        if (last_word!=-1) neu0_ac[last_word]=0;  //delete previous activation

////        last_word=word;

////        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
////        history[0]=last_word;

////    if (independent && (word==0)) netReset();

////        wordcn++;
////    }
}

void CRnnLM::computeRecurrentLayer_(int i_word)
{
    neu1.ac = syn0h * neu0.ac;
    //neu0.ac.print();
  //  syn0h.print();
    neu1.ac.addMatrixColumn(syn0v, i_word);
    neu1.ac.logisticActivation();
  //      neu1.ac.print();
}

void CRnnLM::computeOutputLayer_()
{
    neu2.ac = syn1 * neu1.ac;
    neu2.ac.softmaxActivation();
}

void CRnnLM::computeErrorOnOutput_(int i_trueWord)
{
    neu2.fastOutputError(i_trueWord);
}

void CRnnLM::computeErrorOnPrevious_(const Layer& i_nextLayer, Matrix& i_synMat, Layer& i_prevLayer)
{
    i_prevLayer.er = i_synMat.transpose() * i_nextLayer.er;
    i_synMat.transpose();
}

void CRnnLM::applyGradient_(double i_lr, const Vector& i_next, const Vector& i_prev, Matrix& i_mat, double beta)
{
    //syn1 += alpha * (neu2.er * neu1.ac.transpose()) - beta2 * syn1;
    i_mat.fastGradUpdate(i_lr, i_next, i_prev, beta);
}

void CRnnLM::applyGradient_(double i_lr, const Vector& i_next, int i_column, Matrix& o_mat, double beta)
{
    //syn1 += alpha * (neu2.er * neu1.ac.transpose()) - beta2 * syn1;
    o_mat.fastGradColumnUpdate(i_lr, i_next, i_column, beta);
}

void CRnnLM::addGradient_(const Matrix& i_update, Matrix& o_mat , double beta2)
{
    // syn0h += bptt_syn0h - syn0h * beta2;
    o_mat.addExpression(i_update, beta2);
}

void CRnnLM::applyGradient_(const Matrix& i_update, int i_updateColumn, Matrix& io_target, int i_targetColumn, double i_beta)
{
     //syn0v.col(bptt_history[step]) += bptt_syn0v.col(bptt_history[step]) - syn0v.col(bptt_history[step])*beta2;
    io_target.addColumnToColumn(i_targetColumn, i_update, i_updateColumn, i_beta);
}
