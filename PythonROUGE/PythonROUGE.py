"""
Created on Mon Aug 13 10:31:58 2012
Created by Miguel Almeida, mba@priberam.pt
Modified by Brandon Sim to fit with code framework
"""

import os
import re

# Wrapper function to use ROUGE from Python easily
# Inputs:
    # guess_summ_list, a string with the absolute path to the file with your guess summary
    # ref_summ_list, a list of lists of paths to multiple reference summaries.
    # IMPORTANT: all the reference summaries must be in the same directory!
    # (optional) ngram_order, the order of the N-grams used to compute ROUGE
    # the default is 1 (unigrams)
# Output: a tuple of the form (recall,precision,F_measure)
#
# Example usage: PythonROUGE('/home/foo/my_guess_summary.txt',[/home/bar/my_ref_summary_1.txt,/home/bar/my_ref_summary_2.txt])
def PythonROUGE(guess_summ_list,ref_summ_list,ngram_order=3):
    """ Wrapper function to use ROUGE from Python easily. """

    # even though we ask that the first argument is a list,
    # if it is a single string we can handle it
    if type(guess_summ_list) == str:
        temp = list()
        temp.append(ref_summ_list)
        guess_summ_list = temp
        del temp
    
    # even though we ask that the second argument is a list of lists,
    # if it is a single string we can handle it
    if type(ref_summ_list[0]) == str:
        temp = list()
        temp.append(ref_summ_list)
        ref_summ_list = temp
        del temp
    
    # this is the path to your ROUGE distribution
    #ROUGE_path = "C:\Users\BKS\Documents\PythonROUGE\RELEASE-1.5.5\ROUGE-1.5.5.pl"
    #data_path = 'C:\Users\BKS\Documents\PythonROUGE\RELEASE-1.5.5\data'
    ROUGE_path = "RELEASE-1.5.5/ROUGE-1.5.5.pl"
    data_path = "RELEASE-1.5.5/data"
    
    # these are the options used to call ROUGE
    # feel free to edit this is you want to call ROUGE with different options

    # commented out for ROUGE-L
    options = '-a -m -n ' + str(ngram_order) #Rouge-N
    #options = '-a ' # only Rouge-L
    # this is a temporary XML file which will contain information
    # in the format ROUGE uses
    xml_path = 'temp.xml'
    xml_file = open(xml_path,'w')
    xml_file.write('<ROUGE-EVAL version="1.0">\n')
    for guess_summ_index,guess_summ_file in enumerate(guess_summ_list):
        xml_file.write('<EVAL ID="' + str(guess_summ_index+1) + '">\n')
        create_xml(xml_file,guess_summ_file,ref_summ_list[guess_summ_index])
        xml_file.write('</EVAL>\n')
    xml_file.write('</ROUGE-EVAL>\n')
    xml_file.close()
    
    
    # this is the file where the output of ROUGE will be stored
    ROUGE_output_path = 'ROUGE_result.txt'
    
    # this is where we run ROUGE itself
    exec_command = ROUGE_path + ' -e ' + data_path + ' ' + options + ' -x ' + xml_path + ' > ' + ROUGE_output_path
    #exec_command = ROUGE_path + ' -e ' + data_path + ' ' + options + xml_path + ' > ' + ROUGE_output_path #calculate Rouge-L
    print exec_command
    os.system(exec_command)
    
    # here, we read the file with the ROUGE output and
    # look for the recall, precision, and F-measure scores
    recall_list = list()
    precision_list = list()
    F_measure_list = list()
    ROUGE_output_file = open(ROUGE_output_path,'r')
    for n in xrange(ngram_order):
        ROUGE_output_file.seek(0)
        for line in ROUGE_output_file:
            #print line
            match = re.findall('X ROUGE-' + str(n+1) + ' Average_R: ([0-9.]+)',line) # commented out for ROUGE-L
            #match = re.findall('X ROUGE-L Average_R: ([0-9.]+)', line)
            if match != []:
                recall_list.append(float(match[0]))
            match = re.findall('X ROUGE-' + str(n+1) + ' Average_P: ([0-9.]+)',line) # commented out for ROUGE-L
            #match = re.findall('X ROUGE-L Average_P: ([0-9.]+)', line)
            if match != []:
                precision_list.append(float(match[0]))
            match = re.findall('X ROUGE-' + str(n+1) + ' Average_F: ([0-9.]+)',line) # commented out for ROUGE-L
            #match = re.findall('X ROUGE-L Average_F: ([0-9.]+)', line)
            if match != []:
                F_measure_list.append(float(match[0]))
    ROUGE_output_file.close()
    
    # remove temporary files which were created
    os.remove(xml_path)
    os.remove(ROUGE_output_path)

    return (recall_list,precision_list,F_measure_list)
    
    
# This is an auxiliary function
# It creates an XML file which ROUGE can read
# Don't ask me how ROUGE works, because I don't know!
def create_xml(xml_file,guess_summ_file,ref_summ_list):
    xml_file.write('<PEER-ROOT>\n')
    guess_summ_dir = os.path.dirname(guess_summ_file)
    xml_file.write(guess_summ_dir + '\n')
    xml_file.write('</PEER-ROOT>\n')
    xml_file.write('<MODEL-ROOT>\n')
    ref_summ_dir = os.path.dirname(ref_summ_list[0] + '\n')
    xml_file.write(ref_summ_dir + '\n')
    xml_file.write('</MODEL-ROOT>\n')
    xml_file.write('<INPUT-FORMAT TYPE="SPL">\n')
    xml_file.write('</INPUT-FORMAT>\n')
    xml_file.write('<PEERS>\n')
    guess_summ_basename = os.path.basename(guess_summ_file)
    xml_file.write('<P ID="X">' + guess_summ_basename + '</P>\n')
    xml_file.write('</PEERS>\n')
    xml_file.write('<MODELS>')
    letter_list = ['A','B','C','D','E','F','G','H','I','J']
    for ref_summ_index,ref_summ_file in enumerate(ref_summ_list):
        ref_summ_basename = os.path.basename(ref_summ_file)
        xml_file.write('<M ID="' + letter_list[ref_summ_index] + '">' + ref_summ_basename + '</M>\n')
    
    xml_file.write('</MODELS>\n')
    
# This is only called if this file is executed as a script.
# It shows an example of usage.
if __name__ == '__main__':
    #guess_summary_list = ['Example/Guess_Summ_1.txt','Example/Guess_Summ_2.txt']
    #ref_summ_list = [['Example/Ref_Summ_1_1.txt','Example/Ref_Summ_1_2.txt'] , ['Example/Ref_Summ_2_1.txt','Example/Ref_Summ_2_2.txt','Example/Ref_Summ_2_3.txt']]
    
    algos = ['LSA', 'SumBasic']
    for algo in algos:
        for i in range(20):
            fname = 'Tests/' + algo + str(i+1) + '.txt'
            human = 'Tests/Human' + str(i+1) + '.txt'
            guess_summary_list = [fname]
            ref_summ_list = [human]

            recall_list,precision_list,F_measure_list = PythonROUGE(guess_summary_list,ref_summ_list)
            print guess_summary_list[0]
            print 'recall = ' + str(recall_list)
            print 'precision = ' + str(precision_list)
            print 'F = ' + str(F_measure_list)

            with open('Tests/ROUGE_results_out.txt', 'a') as f:
                f.write(guess_summary_list[0] + '\n')
                #f.write(str(recall_list[0]) + ' ' + str(precision_list[0]) + ' ' + str(F_measure_list[0]) + '\n')
                f.write(str(recall_list[0]) + ' ' + str(precision_list[0]) + ' ' + str(F_measure_list[0]) + ' ')
                f.write(str(recall_list[1]) + ' ' + str(precision_list[1]) + ' ' + str(F_measure_list[1]) + ' ')
                f.write(str(recall_list[2]) + ' ' + str(precision_list[2]) + ' ' + str(F_measure_list[2]) + '\n')
                #f.write('recall = ' + str(recall_list) + '\n')
                #f.write('precision = ' + str(precision_list) + '\n')
                #f.write('F = ' + str(F_measure_list) + '\n')