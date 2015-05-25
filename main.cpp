#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include "rnnlmlib.h"


using namespace std;
using namespace RNNLM;

int argPos(char *str, int argc, char **argv)
{
    int a;
    
    for (a=1; a<argc; a++) if (!strcmp(str, argv[a])) return a;
    
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    
    int debug_mode=1;
    
    FileTypeEnum fileformat=TEXT;
    
    int train_mode=0;
    int valid_data_set=0;
    int test_data_set=0;
    int rnnlm_file_set=0;
    
    int alpha_set=0, train_file_set=0;
    
    float lambda=0.75;
    float gradient_cutoff=15;
    float dynamic=0;
    float starting_alpha=0.1;
    float regularization=0.0000001;
    float min_improvement=1.003;
    int hidden_size=30;
    int bptt=0;
    int bptt_block=10;
    int gen=0;
    int independent=0;
    int rand_seed=1;
    int nbest=0;
    int one_iter=0;
    int gpu = 0;
    int test = 0;

    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    
    FILE *f;
    
    if (argc==1) {
    	//printf("Help\n");

    	printf("Recurrent neural network based language modeling toolkit v 0.3d\n\n");

    	printf("Options:\n");

    	//               
    	printf("Parameters for training phase:\n");


    	printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train rnnlm model\n");

        printf("\t-class <int>\n");
        printf("\t\tWill use specified amount of classes to decompose vocabulary; default is 100\n");

        printf("\t\tThis will use old algorithm to compute classes, which results in slower models but can be a bit more precise\n");

    	printf("\t-rnnlm <file>\n");
        printf("\t\tUse <file> to store rnnlm model\n");
        
        printf("\t-binary\n");
        printf("\t\tRnnlm model will be saved in binary format (default is plain text)\n");

    	printf("\t-valid <file>\n");
    	printf("\t\tUse <file> as validation data\n");

    	printf("\t-alpha <float>\n");
    	printf("\t\tSet starting learning rate; default is 0.1\n");
    	
    	printf("\t-beta <float>\n");
    	printf("\t\tSet L2 regularization parameter; default is 1e-7\n");

    	printf("\t-hidden <int>\n");
    	printf("\t\tSet size of hidden layer; default is 30\n");
    	
    	printf("\t-bptt <int>\n");
    	printf("\t\tSet amount of steps to propagate error back in time; default is 0 (equal to simple RNN)\n");
    	
    	printf("\t-bptt-block <int>\n");
    	printf("\t\tSpecifies amount of time steps after which the error is backpropagated through time in block mode (default 10, update at each time step = 1)\n");

    	printf("\t-min-improvement <float>\n");
    	printf("\t\tSet minimal relative entropy improvement for training convergence; default is 1.003\n");

    	printf("\t-gradient-cutoff <float>\n");
    	printf("\t\tSet maximal absolute gradient value (to improve training stability, use lower values; default is 15, to turn off use 0)\n");
    	
    	printf("\t-independent\n");
    	printf("\t\tWill erase history at end of each sentence (if used for training, this switch should be used also for testing & rescoring)\n");
    	printf("\n");

    	return 0;	//***
    }

    
    //set debug mode
    i=argPos((char *)"-debug", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: debug mode not specified!\n");
            return 0;
        }

        debug_mode=atoi(argv[i+1]);

	if (debug_mode>0)
        printf("debug mode: %d\n", debug_mode);
    }

    
    //search for train file
    i=argPos((char *)"-train", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: training data file not specified!\n");
            return 0;
        }

        strcpy(train_file, argv[i+1]);

	if (debug_mode>0)
        printf("train file: %s\n", train_file);

        f=fopen(train_file, "rb");
        if (f==NULL) {
            printf("ERROR: training data file not found!\n");
            return 0;
        }

        train_mode=1;
        
        train_file_set=1;
    }


    //set one-iter
    i=argPos((char *)"-one-iter", argc, argv);
    if (i>0) {
        one_iter=1;

        if (debug_mode>0)
        printf("Training for one iteration\n");
    }
    
    
    //search for validation file
    i=argPos((char *)"-valid", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: validation data file not specified!\n");
            return 0;
        }

        strcpy(valid_file, argv[i+1]);

        if (debug_mode>0)
        printf("valid file: %s\n", valid_file);

        f=fopen(valid_file, "rb");
        if (f==NULL) {
            printf("ERROR: validation data file not found!\n");
            return 0;
        }

        valid_data_set=1;
    }
    
    if (train_mode && !valid_data_set) {
	if (one_iter==0) {
	    printf("ERROR: validation data file must be specified for training!\n");
    	    return 0;
    	}
    }
    
    //search for test file
    i=argPos((char *)"-test", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: validation data file not specified!\n");
            return 0;
        }

        strcpy(test_file, argv[i+1]);

        if (debug_mode>0)
        printf("test file: %s\n", test_file);

        f=fopen(test_file, "rb");
        if (f==NULL) {
            printf("ERROR: test data file not found!\n");
            return 0;
        }

        test=1;
    }

    //set nbest rescoring mode
    i=argPos((char *)"-nbest", argc, argv);
    if (i>0) {
	nbest=1;
        if (debug_mode>0)
        printf("Processing test data as list of nbests\n");
    }

    //set lambda
    i=argPos((char *)"-lambda", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: lambda not specified!\n");
            return 0;
        }

        lambda=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Lambda (interpolation coefficient between rnnlm and other lm): %f\n", lambda);
    }
    
    
    //set gradient cutoff
    i=argPos((char *)"-gradient-cutoff", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: gradient cutoff not specified!\n");
            return 0;
        }

        gradient_cutoff=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Gradient cutoff: %f\n", gradient_cutoff);
    }
    
    
    //set dynamic
    i=argPos((char *)"-dynamic", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: dynamic learning rate not specified!\n");
            return 0;
        }

        dynamic=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Dynamic learning rate: %f\n", dynamic);
    } 
    
    //set independent
    i=argPos((char *)"-independent", argc, argv);
    if (i>0) {
        independent=1;

        if (debug_mode>0)
        printf("Sentences will be processed independently...\n");
    }

    
    //set learning rate
    i=argPos((char *)"-alpha", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: alpha not specified!\n");
            return 0;
        }

        starting_alpha=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Starting learning rate: %f\n", starting_alpha);
        alpha_set=1;
    }
    
    
    //set regularization
    i=argPos((char *)"-beta", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: beta not specified!\n");
            return 0;
        }

        regularization=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Regularization: %f\n", regularization);
    }
    
    
    //set min improvement
    i=argPos((char *)"-min-improvement", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: minimal improvement value not specified!\n");
            return 0;
        }

        min_improvement=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Min improvement: %f\n", min_improvement);
    }

    //set gpu device
    i=argPos((char *)"-gpu", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: gpu id not specified!\n");
            return 0;
        }

        gpu=atof(argv[i+1]);

        if (debug_mode>0)
        printf("Device : %f\n", gpu);
    }


    //set hidden layer size
    i=argPos((char *)"-hidden", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: hidden layer size not specified!\n");
            return 0;
        }

        hidden_size=atoi(argv[i+1]);

        if (debug_mode>0)
        printf("Hidden layer size: %d\n", hidden_size);
    }

    //set bptt
    i=argPos((char *)"-bptt", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: bptt value not specified!\n");
            return 0;
        }

        bptt=atoi(argv[i+1]);
        bptt++;
        if (bptt<1) bptt=1;

        if (debug_mode>0)
        printf("BPTT: %d\n", bptt-1);
    }

    
    //set bptt block
    i=argPos((char *)"-bptt-block", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: bptt block value not specified!\n");
            return 0;
        }

        bptt_block=atoi(argv[i+1]);
        if (bptt_block<1) bptt_block=1;

        if (debug_mode>0)
        printf("BPTT block: %d\n", bptt_block);
    }
    
        
    //set random seed
    i=argPos((char *)"-rand-seed", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: Random seed variable not specified!\n");
            return 0;
        }

        rand_seed=atoi(argv[i+1]);

        if (debug_mode>0)
        printf("Rand seed: %d\n", rand_seed);
    }
    
    //search for binary option
    i=argPos((char *)"-binary", argc, argv);
    if (i>0) {
        if (debug_mode>0)
        printf("Model will be saved in binary format\n");

        fileformat=BINARY;
    }
    
    //search for rnnlm file
    i=argPos((char *)"-rnnlm", argc, argv);
    if (i>0) {
        if (i+1==argc) {
            printf("ERROR: model file not specified!\n");
            return 0;
        }

        strcpy(rnnlm_file, argv[i+1]);

        if (debug_mode>0)
        printf("rnnlm file: %s\n", rnnlm_file);

        f=fopen(rnnlm_file, "rb");

        rnnlm_file_set=1;
    }
    if (train_mode && !rnnlm_file_set) {
    	printf("ERROR: rnnlm file must be specified for training!\n");
    	return 0;
    }
    if (test_data_set && !rnnlm_file_set) {
    	printf("ERROR: rnnlm file must be specified for testing!\n");
    	return 0;
    }
    if (!test_data_set && !test && !train_mode && gen==0) {
    	printf("ERROR: training or testing must be specified!\n");
    	return 0;
    }
    
    
    srand(1);

    cudaSetDevice(gpu);

    CRnnLM<RnnlmRussianMorphology> trainer;

    trainer.setLearningRate(starting_alpha);
    trainer.setGradientCutoff(gradient_cutoff);
    trainer.setRegularization(regularization);
    trainer.setMinImprovement(min_improvement);
    trainer.setRandSeed(rand_seed);
    trainer.setDebugMode(debug_mode);
    trainer.alpha_set=alpha_set;
    trainer.train_file_set=train_file_set;

    ModelOptions options(hidden_size, bptt, bptt_block);

    if(test == 0)
    trainer.trainNet(train_file, valid_file, rnnlm_file, options);
    else
    {
        trainer.testNet(test_file,rnnlm_file,false);
    }
    
    return 0;
}
