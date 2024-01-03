#!/usr/bin/python
#-- coding:utf8 --

with open('dev_en.txt', 'r') as fa:  # 读取需要拼接的前面那个TXT

    with open('dev_zh.txt', 'r') as fb:  # 读取需要拼接的后面那个TXT

        with open('dev.txt', 'w') as fc:  # 写入新的TXT

            for line in fa:

                fc.write(line.strip('\r\n'))  # 用于移除字符串头尾指定的字符
                fc.write('\t')
                temp=fb.readline().replace('（鼓掌）', '')
                temp=temp.replace('（鼓掌声）', '')
                temp=temp.replace('（众人鼓掌）', '')
                temp=temp.replace('（热烈鼓掌）', '')
                temp=temp.replace('（观众鼓掌）', '')
                temp = temp.replace('（观众掌声）', '')

                fc.write(temp)