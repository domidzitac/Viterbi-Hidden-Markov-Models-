{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf600
{\fonttbl\f0\fnil\fcharset0 LucidaGrande;\f1\fnil\fcharset0 LucidaGrande-Bold;}
{\colortbl;\red255\green255\blue255;\red26\green26\blue26;\red255\green255\blue255;}
{\*\expandedcolortbl;;\cssrgb\c13333\c13333\c13333;\cssrgb\c100000\c100000\c100000;}
{\*\listtable{\list\listtemplateid1\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{square\}}{\leveltext\leveltemplateid1\'01\uc0\u9642 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid1}
{\list\listtemplateid2\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{square\}}{\leveltext\leveltemplateid101\'01\uc0\u9642 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid2}
{\list\listtemplateid3\listhybrid{\listlevel\levelnfc23\levelnfcn23\leveljc0\leveljcn0\levelfollow0\levelstartat1\levelspace360\levelindent0{\*\levelmarker \{square\}}{\leveltext\leveltemplateid201\'01\uc0\u9642 ;}{\levelnumbers;}\fi-360\li720\lin720 }{\listname ;}\listid3}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}{\listoverride\listid3\listoverridecount0\ls3}}
\paperw11900\paperh16840\margl1440\margr1440\vieww18320\viewh11460\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
I deeply apologise for the late submission!\
\
Here is how to run my training algorithm:\cb1 \
\
Version1:\
\
\cb3 python3 did233_trainHMM_HW3.py WSJ_23.words 
\f1\b SMALL
\f0\b0  >submission.pos\
\

\f1\b OR
\f0\b0 \
\
Version2 (the one on gradescope):\
\
python3 did233_trainHMM_HW3.py WSJ_23.words 
\f1\b HEURISTIC
\f0\b0  >submission.pos\
\
\
\
These two are because I implemented the first 2 strategies for OOV from the Homework:\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf0 \cb1 \kerning1\expnd0\expndtw0 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0
\f1\b \cf0 SMALL:
\f0\b0 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls1\ilvl0\cf0 \expnd0\expndtw0\kerning0
\'93use 1/1000 (or other number) as your likelihood for all OOV items and use this same likelihood for all parts of speech -- effectively only use the transition probability for OOV words\'94 - 
\f1\b I used SMALL_NUMBER = 1e-5
\f0\b0 \
\pard\tx566\pardeftab720\partightenfactor0
\cf0 \

\f1\b HEURISTIC:
\f0\b0 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf0 \'93make up some hard coded probabilities based on features like capitalization and endings of words, e.g., ends_in_s --> .5 NNS, begins with capital letter --> .5 NNP, nonletter/nonnumber --> .5 punctuation tag, a token consisting of digits --> 1.0 CD, All other probabilities for OOV are 1/1000.\'94\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0
\f1\b \cf0 I made up some hard coded probabilities for example, if the word ends in s/es and is NNS return 0.6; if it ends in d/ed and is VBD/VBN/JJ return 0.6, etc. else I return SMALL_NUMBER = 1e-5\
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls2\ilvl0\cf2 \cb3 \
\pard\tx220\tx720\pardeftab720\li720\fi-720\partightenfactor0
\ls3\ilvl0
\f0\b0 \cf2 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 To test the score use:\
\
\cb3 python3 score.py WSJ_24.pos submission.pos\
\

\f1\b I obtained the following performances:\
-with the SMALL strategy for OOV: 93.26%\
-with the HEURISTIC strategy for OOV: 93.87% (the one in gradescope)
\f0\b0 \cb1 \
\
The functions are well explained in the source code.\
}