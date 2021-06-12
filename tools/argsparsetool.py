import argparse

def getparse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--lstm_hidden',default=512,type=int,help='lstm_hidden')
    parse.add_argument('--lstm_num_layers',default=2,type=int,help='lstm_layers')
    parse.add_argument("--GPUNUM",default = 0,type=int)
    parse.add_argument("--data_path",default='data/Xeon3NLP_round1_train_20210524.txt',type=str)
    parse.add_argument('--num_train',default=18000,type=int)
    parse.add_argument("--bert_path",default='bert',type=str)
    parse.add_argument('--max_length',default=50,type=int)
    parse.add_argument('--hidden',default=1024,type=int,help='bert_hidden')
    parse.add_argument("--lr",default = 8e-6,help='learning_rate' )
    parse.add_argument('--batch_size',default=32,type=int)
    parse.add_argument('--test_path',default='data/Xeon3NLP_round1_test_20210524.txt')

    return parse
