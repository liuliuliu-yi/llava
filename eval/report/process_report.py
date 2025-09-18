input_file = '/data_C/sdb1/lyi/ECG-Chat-master/llava/llava/eval/report_test.csv'
output_file = '/data_C/sdb1/lyi/ECG-Chat-master/llava/llava/eval/report_test_before_rr.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        if 'RR:' in line:
            rr_index = line.index('RR:')
            new_line = line[:rr_index].strip()
            outfile.write(new_line + '\n')
        else:
            outfile.write(line.strip() + '\n')