import os
import csv
import re
import datetime
import pandas
from collections import Counter
import difflib
from datetime import date, timedelta
import matplotlib.pyplot as plt
import string
from itertools import groupby
from collections import OrderedDict, defaultdict
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from jinja2 import Template, Environment, FileSystemLoader
import math
from langdetect import detect_langs
import nltk
from nltk.collocations import *
import pdfkit


# script to read-in list of words
filename = "/Users/joeDiHare/Documents/chat.txt"
filename = "/Users/joeDiHare/Documents/chat_full.txt"
filename = "/Users/joeDiHare/Documents/chatMultUsr.txt"
# 11/10/14, 18:27:25: Nahnahnah Chill: How was your daaaaaay!?
# 06/01/2015, 12:43 - stefano cosentino: Remember the jamper?

# import codecs
# csvReader = csv.reader(codecs.open(filename, 'rU', 'utf-16'))

# with open(filename, newline='',encoding='UTF8') as inputfile:
#     for line in inputfile:
#         print(line)

# detect format from first row
with open(filename, newline='',encoding='UTF8') as inputfile:
    reader = csv.reader(inputfile)
    row1 = next(reader)
if list(filter(None, row1[1].split(' ')))[0].count(':')==1: # EU format
    PartChar = ' - '
    MsgErr1 = ['Messages you send to this chat and calls are now secured with end-to-end encryption. Tap for more info.',
               'Messages you send to this group are now secured with end-to-end encryption. Tap for more info.']
    MsgErr2 = [' changed the subject from ',
               'You added',
               " deleted this group's icon",
               " changed this group's icon",
               'You created group ']
    MsgErr3 = ['<Media omitted>']

else:                                                       # US format
    PartChar = ' '
    MsgErr1 = ['Messages you send to this chat and calls are now secured with end-to-end encryption.',
               'Messages you send to this group are now secured with end-to-end encryption.','Missed Call']
    MsgErr2 = [' changed the subject from ',
               'You added',
               " deleted this group's icon",
               " changed this group's icon",
               'You created group ']
    MsgErr3 = ['<video omitted>', '<image omitted>', '<audio omitted>']

print("Reading chat conversation & first data validation... ", end="")
results, mediaSender, mediaCaption, R = [], [], [], []
with open(filename, newline='',encoding='UTF8') as inputfile:
    # for row in csv.reader(inputfile):
    for row in inputfile:
        row = row.split(",",1)
        R.append(row)
        for elem in range(0,len(row)): # strip all unicode elements
            row[elem] = row[elem].replace('\\','').encode('ascii', 'ignore').decode('unicode_escape').strip()
        if row!=[]: #ignore empty lines
            if re.search(r'(^\d{1,2}/\d{1,2}/\d{2,4}$)', row[0]) is None: # if not a new sender, attach to previous
                # prevent links that have dates in it to be caught
                results[-1][-1] = results[-1][-1] + ' ' + row[0]
            elif all([row[1].partition(PartChar)[-1].strip()!=_msg_err for _msg_err in MsgErr1] + # -rm error messages 1 & 2
                     [_msg_err not in row[1].partition(PartChar)[-1].strip() for _msg_err in MsgErr2]):
                if any([row[1].partition(PartChar)[-1].partition(':')[-1].strip()==_msg_err
                        for _msg_err in MsgErr3]):
                    # count <Media omitted> and remove
                    mediaSender.append(row[1].partition(PartChar)[-1].partition(':')[0].strip())
                    # mediaCaption.append(row[1].partition(':')[-1].partition(':')[-1].strip()) not correct if more than one : in the body
                else:
                    results.append(row)
print('[done]')

# check if EU or US time format and CONVERT to EU time format
format_time = "%d/%m/%Y %H:%M"
for n in results:
    first = n[0].partition('/')[0]
    secnd = n[0].partition('/')[2].partition('/')[0]
    if int(secnd)>12 or len(first)<2:
        # detected format_time = "%m/%d/%Y %H:%M"
        # CONVERT DATE FORMAT
        for nn in range(0,len(results)):
            results[nn][0] = datetime.datetime.strptime(results[nn][0], '%m/%d/%y').strftime('%d/%m/%Y')
        break

print("Create full message lists... ", end="")
bodyraw, dates, body, datesLong, message, tm, sender = [],[],[],[],[],[],[]
for item in results:
    match_date = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', ' '.join(item))
    match_time = re.search(r'(\d+:\d+)', item[1])
    datesLong.append(datetime.datetime.strptime(match_date.group(1) + ' ' + match_time.group(1), format_time))
    dates.append(item[0])
    tm.append(match_time.group(1))
    sender.append(item[1].partition(PartChar)[-1].partition(':')[0].strip())
    message.append(' '.join(item))
    bodyraw.append(item[1])
    body.append(item[1].partition(PartChar)[-1].partition(':')[-1].strip())
print('[done]')

# Detect language
Language = detect_langs(' '.join(body))[0].lang

print("Create conversation lists... ", end="")
# if the previous message is within LONG_BREAK_CON seconds and it is from the same sender, combine them in the same conversation
ConvBody = [body[0]]; ConvSender = [sender[0]]; ConvDates = [dates[0]]; ConvDatesLong = [datesLong[0]]# initialise
ConvMessage = [message[0]]; ConvTime = [tm[0]]; ConvTimeEnd = [tm[0]]
Conversations, LM = [],[]; RT= [['user', -1]]; flag_new_conv=False
bodylast = '(' + sender[0] + ') ' + body[0]
LONG_BREAK_CONV = 60 * 60 # time constant to consider a message as belonging to a new conversation (set to 60 min)
for n in range(1,len(tm)):
    if (datetime.datetime.strptime(tm[n], "%H:%M") - datetime.datetime.strptime(tm[n-1], "%H:%M")).seconds < LONG_BREAK_CONV \
    and sender[n]==sender[n-1]: # same sender, add to last message
        ConvBody[-1] = ConvBody[-1] + '. ' + body[n]
        ConvTimeEnd[-1] = tm[n]
        bodylast = bodylast + '. ' + body[n] if bodylast!='' else  ' ('+sender[n]+') '+body[n]
    else: # break and start new msg in conversation
        ConvBody.append(body[n])
        ConvSender.append(sender[n])
        ConvTimeEnd.append(tm[n])
        ConvTime.append(tm[n])
        ConvDates.append(dates[n])
        ConvMessage.append(message[n])
        ConvDatesLong.append(datesLong[n])
        bodylast = bodylast + ' (' + sender[n] + ') ' +ConvBody[-1]
        if flag_new_conv:
            RT.append([sender[n],
                        round((datetime.datetime.strptime(tm[n],"%H:%M")-datetime.datetime.strptime(ConvTimeEnd[-2],"%H:%M")).seconds/60)])
            flag_new_conv = False
        if (datetime.datetime.strptime(tm[n], "%H:%M") - datetime.datetime.strptime(tm[n-1], "%H:%M")).seconds >= LONG_BREAK_CONV:
            Conversations.append(bodylast)
            bodylast=''
            flag_new_conv = True
print('[done]')


print('\n\n~~~~~~~~~~~~~~~~~~~ DATA ANALYSIS ~~~~~~~~~~~~~~~~~~~~\n')
do_stages = [1,2,3,4,5,6,7,8,9,10]#1,2,3,4,5,6,7,8,9]
# OutputPdf = PdfPages(filename='outputWA.pdf')
users = list(set(sender))
print('Conversations between '+str(len(users))+' users:' + str(users))

# Find first mover (FM) occurrences
users_search = "|".join(users)
FM = []
for item in Conversations:
    FM.append(re.search(users_search, item).group())
FM_counts = Counter(FM)
FM_users = []
for user in users:
    FM_users.append(round(100*FM_counts[user]/sum(Counter(FM_counts).values())))
    print(user + " started a conversation " + str(FM_users[-1]) + "% of the times.")

# Find users' reaction times to initial message in conversation
UsersRT, UsersRTall, UserRTmedian = [], [], []
for user in users:
    tmp=[]
    for item in RT:
        if item[0]==user:
            tmp.append(item[1])
    UsersRTall.append(np.asarray(tmp))
    UsersRT.append(sum(tmp)/len(tmp))
    UserRTmedian.append(np.median(np.asarray(tmp)))
    print('The median reaction time for '+user+' is '+str(UserRTmedian[-1])+' minutes')
# plot histogram of reaction times
# a = np.hstack(UsersRTall[0])
# fig1 = plt.figure(figsize=(6,4))
# plt.hist(a, bins='auto')  # plt.hist passes it's arguments to np.histogram
# plt.show()
# fig1.savefig('RT.png')

indC, ind =[], []
for u in range(0,len(users)):
    indt=[]
    for n in ConvSender:
        indt.append(True) if n==users[u] else indt.append(False)
    indC.append(indt)
    indt=[]
    for n in sender:
        indt.append(True) if n==users[u] else indt.append(False)
    ind.append(indt)

# WHO MESSAGED THE MOST?
if 1 in do_stages:
    message_counts = Counter(ConvSender)
    MsgCountUsr = []
    for user in users:
        MsgCountUsr.append(message_counts[user])
    fig2a = plt.figure(0, figsize=(6,6))
    ax = plt.subplot(111)
    _,ttext,attext = plt.pie(MsgCountUsr, labels=users, autopct='%1.0f%%', startangle=90)#, shadow=True
    # plt.title("Who Messaged the Most?")
    # plt.ylabel("Number of messages")
    for _n in attext:
        _n.set_color('white')
        _n.set_size(40.0)
    # for _n in ttext:
    #     _n.set_size(40.0)
    ax.legend().set_visible(False)
    fig2a.savefig('WhoMessagedTheMost.png')
    # OutputPdf.savefig(fig2a)
    plt.close()

# WHO SENT MORE MEDIA?
if 2 in do_stages:
    mediaSender_counts = Counter(mediaSender)
    MediaCountUsr=[]
    for user in users:
        MediaCountUsr.append(mediaSender_counts[user])
    fig2b = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    plt.bar(range(0,len(users)),MediaCountUsr, width=.5, color=['g','k'])
    ax.set_xlim(0-.5, len(users))
    ax.set_xticks(range(0,len(users)))
    ax.set_xticklabels(users)
    # plt.title("Who sent more media messages?")
    # plt.xlabel("Number of media exchanged")
    plt.ylabel("Number of media exchanged")
    fig2b.savefig('WhoSentMoreMedia.png')
    # OutputPdf.savefig(fig2b)
    plt.close()

    for u in range(0,len(users)):
        print(user+' sent '+str(MsgCountUsr[u])+' messages and '+str(MediaCountUsr[u])+' images/videos.')

# MOST COMMON N WORDS PER USER
circle_mask = np.array(Image.open("circle-mask.png"))
# stwords = set(STOPWORDS)
# stwords.add("said")
# if 3 in do_stages: #  dep on stage 1 and 2
# script to read-in strop words
filename = 'stopwords_' + Language + '.txt'; stopwords = []
with open(filename, newline='',encoding='UTF8') as inputfile:
    for row in csv.reader(inputfile):
        stopwords.append(row[0].lower())
NoWrdsUsr, bodyUsr, bodyCompact, count = [], [], [], []
punctuation = set(string.punctuation)
for u in range(0,len(users)):
    bodyUsr.append([body[i] for i, x in enumerate(ind[u]) if x])
    NoWrdsUsr.append(len(''.join([body[i] for i, x in enumerate(ind[u]) if x]).split(' ')))

    print('\nWord frequency Analysis for user: ' + users[u] )
    count.append(Counter(word for word in ' '.join(bodyUsr[u]).lower().split() if word not in stopwords).most_common(20))
    print(count[u])
    s = ''.join(ch for ch in ' '.join(bodyUsr[u]).lower() if ch not in punctuation)
    bodyCompact.append(s.split()) #compact version of body, all in one string

    #Wordles
    text_user = ' '.join(bodyUsr[u]).lower()
    wc = WordCloud(max_font_size=40, relative_scaling=.5, background_color="white",
                   max_words=50, stopwords=set(stopwords), mask=circle_mask).generate(text_user)#mask=alice_mask,
    fig3 = plt.figure(figsize=(5,5))
    plt.imshow(wc)
    plt.axis("off")
    fig3.savefig(users[u] + '_wordle.png')
    # OutputPdf.savefig(fig3)
    plt.close()

# COUNT STRETCHED WORDS
CountStretchedWrds, StretchedWordsUsr, CountStretchedWrdsRatio = [],[],[]
for u in range(0,len(users)):
    text = ' '.join(bodyUsr[u]).translate(str.maketrans('().;!?', '      ')).lower()
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = text.replace('0123456789','')
    str_conv = list(filter(None, text.split(' ')))  # remove empty strings
    tmp_counter = 0
    StretchedWords = []
    for check_string in str_conv:
        tmp_counter_word = 0
        for c in range(1,len(check_string)):
            tmp_counter_word = tmp_counter_word+1 if check_string[c]==check_string[c-1] else 0
            if tmp_counter_word > 2:
                tmp_counter = tmp_counter + 1
                StretchedWords.append(check_string)
                break
    StretchedWordsUsr.append(', '.join(item[0] for item in Counter(StretchedWords).most_common(5)))
    CountStretchedWrds.append(tmp_counter)
    CountStretchedWrdsRatio.append(round(tmp_counter/len(str_conv),2))
    print(users[u]+" stretched words "+str(CountStretchedWrdsRatio[u]) +
          "% of the times, and the most frequent words were: " + StretchedWordsUsr[u])

# HOW MANY JINX?
if 4 in do_stages:
    jinxNo = []
    for u in range(0,len(users)):
        jinxNo.append(len(difflib.get_close_matches('jinx', ' '.join(bodyUsr[u]).lower().split(), n=100, cutoff=.8)))
        print(users[u] + ' jinxed ' + str(jinxNo[u]) + ' times.')
    # hello - helo - deletes
    # hello - helol - transpose
    # hello - hallo - replaces
    # hello - heallo - inserts
# alphabet = string.ascii_lowercase
# alphabet = 'jainx'
# word = 'jinx'
# splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
# deletes = [a + b[1:] for a, b in splits if b]
# transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
# replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
# inserts = [a + c + b for a, b in splits for c in alphabet]
# match_jinx =list(set([word] + deletes + transposes + replaces + inserts))
# for word in ' '.join(bodyUsr[0]).lower().split():
#     if len(word)>2:
#         for matchj in match_jinx:
#             if difflib.get_close_matches(word, matchj, n=1, cutoff=.9):
#                 print(word+': '+matchj)


# HOW MANY 'LOVE' or 'I LOVE YOU'?
if 5 in do_stages:
    noLove, noWhy = [],[]; noHateU, noIloveU = [0]*len(users),[0]*len(users)
    for u in range(0,len(users)):
        noLove.append(bodyCompact[u].count('love'))
        noWhy.append(bodyCompact[u].count('why'))
        for i in range(0,len(bodyCompact[u])-3):
            if bodyCompact[u][i] + ' ' + bodyCompact[u][i + 1] == "love you" \
                    or bodyCompact[u][i] + ' ' + bodyCompact[u][i + 1] == "luv you":
                noIloveU[u] += 1
            if bodyCompact[u][i] + ' ' + bodyCompact[u][i + 1] == "hate you":
                noHateU[u] += 1
        print(users[u] + " used the word 'love' " + str(noLove[u]) + " times, and said 'I love you' "+str(noIloveU[u]) +
              " times, but also 'I hate you' "+str(noHateU[u])+" times.")

# TIME ANALYSIS
# find unique dates with messages
Duniq=['initialize']
for n in ConvDates:
    Duniq.append(n) if Duniq[-1]!=n else ''
Duniq.pop(0)

# Dates for user (indexing, full body)
datesUser=[]
for u in range(0,len(users)):
    tmp = [ConvDates[n] for n in range(0, len(ConvDates)) if ind[u][n]]
    datesUser.append(tmp)

# A list containing all uniques days of the dates from
d1 = date(int(ConvDates[0][-4:10]),int(ConvDates[0][3:5]),int(ConvDates[0][0:2]))
d2 = date(int(ConvDates[-1][-4:10]),int(ConvDates[-1][3:5]),int(ConvDates[-1][0:2]))
dd = [d1 + timedelta(days=x) for x in range((d2-d1).days + 1)]
noMsgPerDay = []
for u in range(0, len(users)):
    tmp = []
    for d in dd:
        tmp.append(ConvDates.count(d.strftime('%d/%m/%Y')))
    noMsgPerDay.append(tmp)

# which months are included
no_months=abs((d1.year - d2.year)*12 + d1.month - d2.month)
# MONTHS=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
years, months, months_short, month_legend = [], [], [], []
for k in range(d1.month,d1.month+no_months):
    months.append(MONTHS[k%12 - 1])
    months_short.append(MONTHS[k%12 - 1][0])
    years.append(d1.year + math.floor(k/12.01))
    month_legend.append(months_short[-1] if k%12!=1 else months[-1]+str(years[-1])[2:4])
    # print(str(months[-1])+' '+str(years[-1]))

# count messages as function of months
current_month = d1.month
tmp, Tmp1, Tmp2, ConvSenderByMonth, CounterSenderByMonth = [], [], [], [], []
for user in users:
    Tmp1, Tmp2 = [], []
    for k in range(0, no_months):
        m = (k+d1.month-1)%12
        tmp = [ConvBody[o] for o in range(0,len(ConvBody))
               if ConvDatesLong[o].month==m+1 and ConvDatesLong[o].year==years[k] and ConvSender[o]==user]
        Tmp1.append(tmp)
        Tmp2.append(len(tmp))
    ConvSenderByMonth.append(Tmp1)
    CounterSenderByMonth.append(Tmp2)
## PLOT Message Distribution over period

if 5 in do_stages:
    cols = [[.5,.5,.8,.3],[.5,.6,.1,.3],'b','y','r','g']
    fig3b = plt.figure(figsize=(10, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom();ax.get_yaxis().tick_left()
    width = 0.35
    for u in range(0,len(users)):
        xdata = np.arange(len(months))+u*width
        ydata = CounterSenderByMonth[u]
        ax.bar(xdata,ydata, width, color=cols[u])
    plt.legend(users)
    plt.xticks(xdata, month_legend, rotation='horizontal')
    # ax.set_xlabel('months', fontsize=16)
    ax.set_ylabel('Number of conversations', fontsize=16)
    # ax.set_title('Message Distribution per month', fontsize=18)
    ax.xaxis.set_tick_params(size=0.2)
    ax.yaxis.set_tick_params(size=0.2)
    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((1,1,1))
    ax.spines['top'].set_color((1,1,1))
    # tweak the axis labels
    max_mocon = max(max(CounterSenderByMonth)) + 5
    ax.set_xlim(0-.1, no_months+.1); xlab = ax.xaxis.get_label(); xlab.set_style('italic')
    ax.set_ylim(0-.1, max_mocon+.1); ylab = ax.yaxis.get_label(); ylab.set_style('italic')
    # tweak the title
    ttl = ax.title;  ttl.set_weight('bold')
    fig3b.savefig('sender_per_month.png')
    # OutputPdf.savefig(fig3b)
    plt.close()


# WHAT DAYS OF THE WEEK DO WE MESSAGE LESS or MORE?
if 6 in do_stages:
    # subplot(1) LESS
    #  extract days with 0 messages
    week=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig4, ax1 = plt.subplots(2, figsize=(6,6), sharex=True)
    cols = [[.5,.5,.8,.3],[.5,.6,.1,.3],'b','y','r','g']
    ddaysCount = []
    for u in range(0, len(users)):
        grouped_L = [[k, sum(1 for i in g)] for k,g in groupby(noMsgPerDay[u])] #sum occurrences of consecutive numbers
        count_noMsg = [grouped_L[n][1] for n in range(0,len(grouped_L)) if grouped_L[n][0] is 0] # only looks at zeros
        ddays = [dd[n].strftime('%a') for n in range(0,len(noMsgPerDay[u])) if noMsgPerDay[u][n] is 0]
        ddaysCount.append(OrderedDict((w, ddays.count(w)) for w in week)) #ddaysCount = {w:ddays.count(w) for w in week}
        # plot
        ax1[0].bar(range(len(ddaysCount[u])), ddaysCount[u].values(), color=cols[u],  width=.5, bottom=ddaysCount[u].values()
        if u > 0 else [0]*7)
        plt.xticks(range(len(ddaysCount[u])), ddaysCount[u].keys())
        plt.ylabel('Occurrence'); plt.title('Number of times there were ZERO :( messages on a specific day of the week')
        plt.legend(users)
    # subplot(2) MORE
    #  WHAT DAYS OF THE WEEK DO WE MESSAGE MORE?
    noMsgPerWeekday=[]
    for u in range(0, len(users)):
        ddays = [[dd[n].strftime('%a'),noMsgPerDay[u][n]] for n in range(0,len(noMsgPerDay[u])) if noMsgPerDay[u][n] > 0]
        ddaysCount = OrderedDict((w, 0) for w in week)
        for d in ddays:
            ddaysCount[d[0]]=ddaysCount[d[0]]+d[1]
        noMsgPerWeekday.append(ddaysCount)
        ax1[1].bar(range(len(noMsgPerWeekday[u])), noMsgPerWeekday[u].values(), color=cols[u],
                   width=.5,bottom=noMsgPerWeekday[u].values() if u > 0 else [0] * 7)
        plt.xticks(range(len(noMsgPerWeekday[u])), noMsgPerWeekday[u].keys())
        plt.ylabel('Occurrence'); plt.title('Messaging during the week')
    fig4.savefig('weekdays.png')
    # OutputPdf.savefig(fig4)
    plt.close()

# WHAT HOUR OF THE DAY WE MESSAGE MORE?
hours=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
hours_labels=['00','','02','','04','','06','','08','','10','','12','','14','','16','','18','','20','','22','']
if 7 in do_stages:
    fig5b = plt.figure(figsize=(6, 6))
    ax2 = plt.subplot(111)
    noMsgPerHour=[]
    width=.3
    for u in range(0, len(users)):
        tmp = []
        for t in hours:
            tmp.append(sum([ConvTime[n][:2].count(t) for n in range(0,len(ConvTime)) if indC[u][n]]))
        noMsgPerHour.append(tmp)
        ax2.bar(u*width+np.arange(0,len(noMsgPerHour[u])), noMsgPerHour[u], color=cols[u], width=width)
        plt.xticks(np.arange(0,len(noMsgPerHour[u])), hours_labels)
    plt.ylabel('Occurrence'); plt.xlabel('Hour of the day'); plt.title('Messaging during the day')
    plt.legend(users)
    fig5b.savefig('dayshour.png')
    # OutputPdf.savefig(fig5b)
    plt.close()

## PLOT Message Distribution over period
if 8 in do_stages:
    fig6 = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom();ax.get_yaxis().tick_left()
    for u in range(0,len(users)):
        xdata = range(0,len(dd))
        ydata1 = noMsgPerDay[u-1] if u>0 else [0]*len(dd)
        ydata2 = [sum(x) for x in zip(noMsgPerDay[u], noMsgPerDay[u-1])] if u>0 else noMsgPerDay[u]
        l = ax.fill_between(xdata,  ydata1, ydata2, facecolor=cols[u], alpha=0.5)
        # change the fill color, edge color and thickness
        # l.set_facecolors([[.5,.5,.8,.3]])
        l.set_edgecolors([[0, 0, .5, .3]])
        l.set_linewidths([.1])
    ax.set_xlabel('Day posting', fontsize=16)
    ax.set_ylabel('Number of messages', fontsize=16)
    ax.set_title('Message Distribution', fontsize=18)
    # set the limits
    ax.set_xlim(0, len(dd))
    ax.set_ylim(0, max(ydata2) + 5)
    # add more ticks
    ax.set_xticks(range(0,len(dd),30))
    # remove tick marks
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    # change the color of the top and right spines to opaque gray
    ax.spines['right'].set_color((1,1,1))
    ax.spines['top'].set_color((1,1,1))
    # tweak the axis labels
    xlab = ax.xaxis.get_label()
    ylab = ax.yaxis.get_label()
    xlab.set_style('italic')
    xlab.set_size(10)
    ylab.set_style('italic')
    ylab.set_size(10)
    # tweak the title
    ttl = ax.title
    ttl.set_weight('bold')
    fig6.savefig('message_distribution.png')
    # OutputPdf.savefig(fig6)
    plt.close()

# most common 3-word sentences
TARGET = ['good night', 'happy birthday', 'happy anniversary']
# script to read-in strop words
filename = 'stopwords_3_words.txt'; stopwords3 = []
with open(filename, newline='', encoding='UTF8') as inputfile:
    for row in csv.reader(inputfile):
        stopwords3.append(row[0].lower())
good_night, happy_bday, happy_anniv = [], [], []
if 9 in do_stages: # depend on stage 3
    ngramsUsr, lensGrams = [], []
    for u in range(0,len(users)):
        text = ' '.join(bodyUsr[u]).translate(str.maketrans('().;!?', '      ')).lower()
        text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
        str_conv = list(filter(None, text.split(' '))) # remove empty strings

        buffer_words2, buffer_words3, buffer_words4 = [], [], []
        for kk in range(4,len(str_conv)):
            buffer_words2.append(str_conv[kk-1] + ' ' + str_conv[kk])
            buffer_words3.append(str_conv[kk - 2] + ' ' + str_conv[kk - 1] + ' ' + str_conv[kk])
            buffer_words4.append(str_conv[kk - 3] + ' ' + str_conv[kk - 2] + ' ' + str_conv[kk - 1] + ' ' + str_conv[kk])

        good_night.append(len([buffer_words2[o] for o in range(0, len(buffer_words2)) if buffer_words2[o] == TARGET[0]]))
        happy_bday.append(len([buffer_words2[o] for o in range(0, len(buffer_words2)) if buffer_words2[o] == TARGET[1]]))
        happy_anniv.append(len([buffer_words2[o] for o in range(0, len(buffer_words2)) if buffer_words2[o] == TARGET[2]]))

        res2 = Counter(buffer_words2).most_common(50)
        res3 = Counter(buffer_words3).most_common(50)
        res4 = Counter(buffer_words4).most_common(50)

        ngramsUsr.append([wrd[0] for wrd in res3 if wrd[0] not in stopwords3])
        lensGrams.append(min(5,len(ngramsUsr[u])))
        print(users[u] + "'s favourite expressions are: " + ' | '.join([ngramsUsr[u][k] for k in range(0,lensGrams[u])]) + ';')

        # FILTER_NO = 3
        # Ngram analysis, but not sure about the results I get...
        # print('\nN=2 grams')
        # bigram_measures = nltk.collocations.BigramAssocMeasures()
        # finder = BigramCollocationFinder.from_words(str_conv)
        # finder.apply_freq_filter(FILTER_NO)
        # res2 = finder.nbest(bigram_measures.pmi, 5)
        # # print('\nN=3 grams')
        # trigram_measures = nltk.collocations.TrigramAssocMeasures()
        # finder = TrigramCollocationFinder.from_words(str_conv)
        # # only bigrams that appear 3+ times
        # finder.apply_freq_filter(FILTER_NO)
        # # return the 10 n-grams with the highest PMI
        # res3 = finder.nbest(trigram_measures.pmi, 5)
        # ngrams = []
        # for item in res3:
        #     tmp = ''
        #     for wrd in item:
        #         tmp = tmp + ' ' + wrd
        #     ngrams.append(tmp)
        # for item in res2:
        #     if not any([set(item) < set(res3[o]) for o in range(0,len(res3))]):
        #         tmp = ''
        #         for wrd in item:
        #             tmp = tmp + ' ' + wrd
        #         ngrams.append(tmp.strip())
        # ngramsUsr.append(ngrams)
        # print(users[u] + "'s favourite expressions are: " + ' | '.join([k for k in ngramsUsr[u]]) + ';')


    # Laughter analysis
if 10 in do_stages:
    LaughterUsr = []
    for u in range(0,len(users)):
        all_words = ' '.join(bodyUsr[u]).split()
        hah = [word for word in all_words if 'hah' in word.lower()]
        LaughterUsr.append([len(hah), len(hah)/len(all_words), len(max(hah, key=len))])
        print(users[u] + ' laughed '+str(LaughterUsr[u][0])+' times (every '+str(round(1/LaughterUsr[u][1]))+' words). '
                         'The longest laugh required ' + str(LaughterUsr[u][2])+' characters!')
    JokeCrakers = []
    for n in range(0,len(ConvBody)):
        if 'haha' in ConvBody[n]:
            if ConvSender[n-1]!=ConvSender[n]:
                JokeCrakers.append(ConvSender[n-1])
            # print(ConvSender[n-1] + ' joked saying: ' + ConvBody[n-1] +
            #       '; and ' + ConvSender[n] + ' responded laughing ('+ ConvBody[n] + ')')
    JokeMade = Counter(JokeCrakers)
    JokeCrakers = []
    for user in users:
        JokeCrakers.append(JokeMade[user])
        print(user + ' made ' + str(JokeMade[user]) + ' jokes that received a laugh.')
    TotLaughs = sum(JokeCrakers)
    JokePerMsgPerc = round(100*TotLaughs/len(ConvBody),1)
    print('There was a total of '+ str(TotLaughs) +' jokes, which amounts to '+str(JokePerMsgPerc)+'% of the conversations.')

    JokeCrakers_best = [users[JokeCrakers.index(max(JokeCrakers))], max(JokeCrakers)]

    fig10 = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    plt.bar(range(0,len(users)),JokeCrakers, width=.5, color=['g','k'])
    ax.set_xlim(0-.5, len(users))
    ax.set_xticks(range(0,len(users)))
    ax.set_xticklabels(users)
    # plt.title("Who make more funny jokes")
    plt.ylabel("Number of jokes")
    fig10.savefig('WhoMadeMoreJokes.png')
    # OutputPdf.savefig(fig2b)
    plt.close()


# Render html file
env = Environment(loader=FileSystemLoader('templates'))
# name_templ_html = 'index_approach1.html'
name_templ_html = 'index_approach2.html'
template = env.get_template(name_templ_html)
output_from_parsed_template = template.render(
    no_users=len(users), do_stages=do_stages, users=users,
    UserRTmedian=UserRTmedian, FM_users=FM_users,
    MsgCountUsr=MsgCountUsr,MediaCountUsr=MediaCountUsr,
    jinxNo=jinxNo,
    noLove=noLove,noIloveU=noIloveU,noHateU=noHateU,
    ngramsUsr=ngramsUsr, lensGrams=lensGrams,
    CountStretchedWrdsRatio=CountStretchedWrdsRatio,StretchedWordsUsr=StretchedWordsUsr,
    happy_bday=happy_bday,happy_anniv=happy_anniv,good_night=good_night,
    JokeCrakers=JokeCrakers,JokeCrakers_best=JokeCrakers_best,TotLaughs=TotLaughs,JokePerMsgPerc=JokePerMsgPerc)
# to save the results
with open("OutputAnalysis.html", "w") as fh:
    fh.write(output_from_parsed_template)

os.system("open OutputAnalysis.html")
#
# import pdfkit
# with open('OutputAnalysis.html') as f:
#     pdfkit.from_file(f, 'out.pdf')

# To do:
# Module to: Check how words are stretched as in informal conversations
# Module to: Find anniversaries by frequencies of "happy birthday"
# Module to: Swear words
# Module to: Implement detection block for US/EU pattern
# Module to: Add more stopwords to preferred sentences
# Module to: Who has is the funniest? count 'haha'
# Module to: Copy html to email and send