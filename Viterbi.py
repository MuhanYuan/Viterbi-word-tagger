import numpy as np
import datetime
import sys

def max_index(l):
    max_value = max(l)
    max_index = l.index(max_value)
    return max_index

def calculate_pr(dic,c2,smooth=False,length = 0):
    sum = 0.0
    for i in dic:
        sum+= dic[i]
    if smooth == False:
        p = dic[c2]/sum
    else:
        if c2 in dic:
            p = (dic[c2]+0.2)/(sum+length)
        else:
            p =  0.2/(sum+length)
    return p


def main():
    start_time = datetime.datetime.now()
    with open(sys.argv[1],"r") as filedata:
        train_datatable = [[i.rsplit('/',1) for i in j.strip().split(" ")] for j in filedata]

    with open(sys.argv[2],"r") as filedata:
        test_datatable = [[i.rsplit('/',1) for i in j.strip().split(" ")] for j in filedata]
    outfile = open("POS.test.out","w")
    bigram_tag_dic = {} # tag1 -- tag2
    word_tag_dic ={}  # tag -- word
    tag_word_dic = {} # word -- tag
    tag_list = []
    word_list = []

    for line in train_datatable:
        previous = -1
        for word_ind in range(len(line)):
            try:
                if previous == -1:
                    pre_tag = "Phi"
                else:
                    pre_tag = line[word_ind-1][1]
                    if "|" in pre_tag:
                        pre_tag = pre_tag.split("|")

                cur_tag = line[word_ind][1]

                if "|" in cur_tag:
                    cur_tag = cur_tag.split("|")
                cur_word = line[word_ind][0]
            except:
                continue

            # bigram_tag_dic
            if type(pre_tag) == type([]):
                for t in pre_tag:
                    if t not in bigram_tag_dic:
                        bigram_tag_dic[t] = {}
                    if type(cur_tag) == type([]):
                        for c_t in cur_tag:
                            if c_t not in bigram_tag_dic[t]:
                                bigram_tag_dic[t][c_t] = 1
                            else:
                                bigram_tag_dic[t][c_t] += 1
                    else:
                        if cur_tag not in bigram_tag_dic[t]:
                            bigram_tag_dic[t][cur_tag] = 1
                        else:
                            bigram_tag_dic[t][cur_tag] += 1
            else:
                if pre_tag not in bigram_tag_dic:
                    bigram_tag_dic[pre_tag] = {}
                if type(cur_tag) == type([]):
                    for c_t in cur_tag:
                        if c_t not in bigram_tag_dic[pre_tag]:
                            bigram_tag_dic[pre_tag][c_t] = 1
                        else:
                            bigram_tag_dic[pre_tag][c_t] += 1
                else:
                    if cur_tag not in bigram_tag_dic[pre_tag]:
                        bigram_tag_dic[pre_tag][cur_tag] = 1
                    else:
                        bigram_tag_dic[pre_tag][cur_tag] += 1

            # word_tag_dic
            if type(cur_tag) == type([]):
                for c_t in cur_tag:
                    if c_t not in word_tag_dic:
                        word_tag_dic[c_t] = {}
                    if cur_word not in word_tag_dic[c_t]:
                        word_tag_dic[c_t][cur_word] = 1
                    else:
                        word_tag_dic[c_t][cur_word] +=1
                    if c_t not in tag_list:
                        tag_list.append(c_t)
            else:
                if cur_tag not in word_tag_dic:
                    word_tag_dic[cur_tag] = {}
                if cur_word not in word_tag_dic[cur_tag]:
                    word_tag_dic[cur_tag][cur_word] = 1
                else:
                    word_tag_dic[cur_tag][cur_word] +=1
                if cur_tag not in tag_list:
                    tag_list.append(cur_tag)

            # tag_word_dic
            if cur_word not in tag_word_dic:
                tag_word_dic[cur_word] = {}
            if type(cur_tag) == type([]):
                for c_t in cur_tag:
                    if c_t not in tag_word_dic[cur_word]:
                        tag_word_dic[cur_word][c_t] = 1
                    else:
                        tag_word_dic[cur_word][c_t] +=1
            else:
                if cur_tag not in tag_word_dic[cur_word]:
                    tag_word_dic[cur_word][cur_tag] = 1
                else:
                    tag_word_dic[cur_word][cur_tag] +=1

            previous = word_ind

            if line[word_ind][0] not in word_list:
                word_list.append(line[word_ind][0])

    for line in test_datatable:
        for word in line:
            if word[0] not in word_list:
                word_list.append(word[0])

    tag_list_len = len(tag_list)
    word_list_len = len(word_list)
    pop_tag = sorted(word_tag_dic.keys(),key = lambda x:len(word_tag_dic[x]),reverse=True)[0]

    count =0
    correct_count = 0
    spe_count = 0
    for line in test_datatable:
        score = []
        backptr = []
        for word_num in range(len(line)):
            temp_score_row = []
            temp_back_row = []
            if word_num == 0:
                # Initialization Step
                for tag in word_tag_dic.keys():
                    temp_value = calculate_pr(word_tag_dic[tag],line[word_num][0],smooth=True,length = word_list_len) * calculate_pr(bigram_tag_dic["Phi"],tag,smooth = True, length = tag_list_len)
                    temp_score_row.append(temp_value)
                    temp_back_row.append(0)
                # score.append(temp_score_row)
                # backptr.append(temp_back_row)
            else:
                # Iteration Step
                for tag1 in word_tag_dic:
                    temp_sel_list = []
                    for index, tag2 in enumerate(word_tag_dic):
                        temp_value = score[word_num-1][index] * calculate_pr(bigram_tag_dic[tag2],tag1,smooth = True, length = tag_list_len)
                        temp_sel_list.append(temp_value)
                    temp_score_row.append(max(temp_sel_list) * calculate_pr(word_tag_dic[tag1],line[word_num][0],smooth=True, length= word_list_len))
                    temp_back_row.append(max_index(temp_sel_list))
            score.append(temp_score_row)
            backptr.append(temp_back_row)

        # Sequence Identification
        line_output = ""
        seq_list=[]
        for w in sorted(range(len(line)),reverse= True):
            if w == len(line)-1:
                tag_ind = max_index(score[w])
            else:
                tag_ind = backptr[w+1][seq_list[-1]]
            seq_list.append(tag_ind)
            tag = word_tag_dic.keys()[tag_ind]
            if tag == line[w][1]:
                correct_count+=1

            # else:
            #     print " ".join([wd[0] for wd in line])
            #     print line[w][0],line[w][1],tag

            count += 1
            line_output = line[w][0]+"/"+tag+" "+line_output
        outfile.write(line_output+"\n")

    print "Viterbi accuracy: " + str(float(correct_count)/count)

    # baseline
    count =0
    correct_count = 0
    for line in test_datatable:
        for w in line:
            try:
                if sorted(tag_word_dic[w[0]], key = lambda x:tag_word_dic[w[0]][x] ,reverse=True)[0] == w[1]:
                    correct_count+=1
            except:
                if pop_tag == w[1]:
                    correct_count+=1
            count+= 1
    print "Baseline Accuracy: "+str(float(correct_count)/count)
    print datetime.datetime.now() - start_time


if __name__=='__main__':
	main()
