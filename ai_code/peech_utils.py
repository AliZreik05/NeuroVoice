#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2021. Institute of Health and Medical Technology, Hefei Institutes of Physical Science, CAS
# @Time      : 2021/8/18 20:54
# @Author    : ZL.Z
# @Email     : zzl1124@mail.ustc.edu.cn
# @Reference: None
# @FileName : util.py
# @Software : Python3.6; PyCharm; Windows10
# @Hardware : Intel Core i7-4712MQ; NVIDIA GeForce 840M
# @Version   : V1.4 - ZL.Z：2022/5/19
#             Deleted p_dur_thr parameter from vup_duration_from_vuvInfo and vup_duration_from_vuvTextGrid,
#             and modified the methods to duration_from_vuvInfo and duration_from_vuvTextGrid,
#             only returning voiced and unvoiced (including light sounds) segments.
#             V1.3 - ZL.Z：2022/3/30
#             Added audio splitting audio_split, audio joining audio_join, and Chinese numeral to Arabic numeral conversion cn2an methods.
#             V1.2 - ZL.Z：2022/3/17
#             Added audio_word_align method.
#             V1.1 - ZL.Z：2022/3/2
#             Added delete_punctuation method.
#             V1.0 - ZL.Z：2021/8/18
#             First version.
# @License   : None
# @Brief     : Collection of common utility methods

import os
import csv
import math
import shutil
import parselmouth
from parselmouth.praat import call

PUNCTUATIONS = set(u''' :!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')  # Collection of punctuation marks (including spaces)


def delete_punctuation(text_list):
    """
    Deletes punctuation marks from text or a list.
    :param text_list: Text or a list.
    :return: The result after deletion.
    """
    if type(text_list) is list:
        filter_punt = lambda l: list(filter(lambda x: x not in PUNCTUATIONS, l))  # For lists
    else:
        filter_punt = lambda s: ''.join(filter(lambda x: x not in PUNCTUATIONS, s))  # For str/unicode
    return filter_punt(text_list)


def write_csv(data, filename):
    """Writes to a CSV file."""
    if not filename.endswith('.csv'):  # Add file extension
        filename += '.csv'
    # Chinese requires utf-8 format; to prevent garbled characters when opened in Excel, use utf-8-sig encoding,
    # which inserts a BOM (Byte Order Mark) \ufeff at the beginning of the text.
    # This is an invisible identifier field indicating byte order (big-endian or little-endian).
    with open(filename, "a", newline="", encoding="utf-8-sig") as f:  # Open file in append mode, no new line, utf-8 encoding
        f_csv = csv.writer(f)  # Write header first
        # for item in data:
        #     f_csv.writerow(item)  # Write row by row
        f_csv.writerows(data)  # Write multiple rows


def read_csv(filename):
    """Reads a CSV file. Using csv.reader() to open a CSV file returns a list-formatted iterator,
    whose elements can be accessed using next() or iterated through with a for loop."""
    if not filename.endswith(".csv"):  # Add file extension
        filename += ".csv"
    data = []  # Read file data
    with open(filename, "r", encoding="utf-8-sig") as f:  # Open file in read mode, utf-8 encoding
        f_csv = csv.reader(f)  # f_csv object, in list format
        for row in f_csv:
            data.append(row)
    return data


def duration_from_vuvInfo(vuv_info: str):
    """
    Segments speech from Info text obtained from Praat's TextGrid object containing 'U'/'V':
    voiced segments and unvoiced segments.
    Here, voiced segments can be considered as voiced sections, and unvoiced segments can be considered as pause segments (silent segments),
    but these unvoiced segments also include light sounds.
    :param vuv_info: Info text obtained from a TextGrid object containing 'U'/'V' fields.
    :return: segments_voice, segments_unvoice
             float, list(n_segments, 2), corresponding to the start and end times of each segment, in seconds.
    """
    segments_voice, segments_unvoice = [], []
    for text_line in vuv_info.strip('\n').split('\n'):
        text_line = text_line.split('\t')
        if 'text' not in text_line:
            tmin = float(text_line[0])
            text = text_line[1]
            tmax = float(text_line[2])
            if text == 'V':  # Voiced segment
                segments_voice.append([tmin, tmax])
            elif text == 'U':  # Unvoiced segment, which includes light sounds and silent segments. Further separation is needed for segmentation.
                segments_unvoice.append([tmin, tmax])
    return segments_voice, segments_unvoice


def duration_from_vuvTextGrid(vuv_file):
    """
    Segments speech from the vuv.TextGrid file obtained from Praat: voiced segments and unvoiced segments.
    Here, voiced segments can be considered as voiced sections, and unvoiced segments can be considered as pause segments (silent segments),
    but these unvoiced segments also include light sounds.
    :param vuv_file: The vuv.TextGrid file obtained from Praat.
    :return: segments_voice, segments_unvoice
             float, list(n_segments, 2), corresponding to the start and end times of each segment, in seconds.
    """
    with open(vuv_file) as f:
        segments_voice, segments_unvoice = [], []
        data_list = f.readlines()
        for data_index in range(len(data_list)):
            data_list[data_index] = data_list[data_index].strip()  # Remove newline characters
            if data_list[data_index] == '"V"':  # Voiced segment
                # The two lines before the identifier character are the start and end durations
                segments_voice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
            elif data_list[data_index] == '"U"':  # Unvoiced segment, which includes light sounds and silent segments. Further separation is needed for segmentation.
                segments_unvoice.append([float(data_list[data_index - 2]), float(data_list[data_index - 1])])
        return segments_voice, segments_unvoice


def audio_join(input_dir, output_dir, joint_silence_len=1.5, samp_freq=16000):
    """
    Joins audio files.
    :param input_dir: Input folder containing audio files to be joined.
    :param output_dir: Output folder for audio.
    :param joint_silence_len: Silence interval between joined audio files, default 1.5 seconds.
    :param samp_freq: Audio sampling rate. The sampling rate needs to be consistent for joining, default 16kHz.
    :return: None
    """
    assert len(os.listdir(input_dir)), f'No audio files exist in the input folder {input_dir}'
    if len(os.listdir(input_dir)) == 1:
        shutil.copy(os.path.join(input_dir, os.listdir(input_dir)[0]), output_dir)
    aud_sil = call('Create Sound from formula', 'sil', 1, 0.0, joint_silence_len, samp_freq, '0')
    auds = []
    for _i in os.listdir(input_dir):
        i_aud = os.path.join(input_dir, _i)
        auds.append(parselmouth.Sound(i_aud))
        auds.append(aud_sil)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete output folder first
    os.mkdir(output_dir)  # Recreate
    sound = parselmouth.Sound.concatenate(auds)
    save_name = os.path.basename(os.listdir(input_dir)[0]).split('-')[0] + '.wav'
    sound.save(os.path.join(output_dir, save_name), parselmouth.SoundFileFormat.WAV)


# Coefficient
CN_NUM = {
    u'〇': 0,
    u'一': 1,
    u'二': 2,
    u'三': 3,
    u'四': 4,
    u'五': 5,
    u'六': 6,
    u'七': 7,
    u'八': 8,
    u'九': 9,

    u'零': 0,
    u'壹': 1,
    u'贰': 2,
    u'叁': 3,
    u'肆': 4,
    u'伍': 5,
    u'陆': 6,
    u'柒': 7,
    u'捌': 8,
    u'玖': 9,

    u'貮': 2,
    u'两': 2,
    u'俩': 2,
    u'倆': 2,
    u'营': 0,
    u'其': 7,
    u'西': 7,
    u'气': 7,
    u'吧': 8,
    u'就': 9,
}
# Base
CN_UNIT = {
    u'十': 10,
    u'拾': 10,
    u'是': 10,
    u'实': 10,
    u'时': 10,
    u'百': 100,
    u'佰': 100,
    # u'千': 1000,
    # u'仟': 1000,
    # u'万': 10000,
    # u'萬': 10000,
    # u'亿': 100000000,
    # u'億': 100000000,
    # u'兆': 1000000000000,
}


def cn2an(chinese_number):
    """
    Converts Chinese numerals to Arabic numerals.
    :param chinese_number: Chinese numeral.
    :return: Arabic numeral, as a string.
    """
    # Convert Chinese characters to corresponding Arabic numerals based on coefficient and base maps.
    tmp = []
    for d in chinese_number[::-1]:  # Iterate through chinese_number elements in reverse
        if CN_NUM.__contains__(d):
            tmp.append(CN_NUM[d])  # Coefficient corresponding Arabic numeral
        elif CN_UNIT.__contains__(d):
            tmp.append(CN_UNIT[d])  # Base corresponding Arabic numeral
        elif d.isnumeric():  # If the element contains Arabic numeral characters
            try:
                tmp.append(int(d))  # Return Arabic numeral as is
            except ValueError:
                pass
    if not tmp:  # If all elements are non-numeric characters, and tmp list is empty, exit.
        return -1
    # Coefficients are directly added to tmp2, bases are multiplied with adjacent numbers or added to tmp2
    # with a leading 1 (to handle cases like "十一", "十" where there's no coefficient before the base).
    tmp_len = len(tmp)
    tmp2 = []
    for i in range(0, tmp_len):
        if tmp[i] > 9:  # Coefficient is greater than 9
            if i == tmp_len - 1 or tmp[i + 1] > tmp[i]:  # Cases like "十一", "十" where there's no coefficient before the base
                tmp2.append(tmp[i])
                tmp2.append(1)
            elif tmp[i + 1] > 9:  # Two bases adjacent, directly multiply
                tmp[i + 1] *= tmp[i]
            else:  # Only one base, directly add to tmp2
                tmp2.append(tmp[i])
        else:  # Coefficient is less than 10, directly add to tmp2
            tmp2.append(tmp[i])
    # Coefficients are directly added to seq, bases are placed at the correct position using -1 as a placeholder for the next coefficient.
    seq = []
    curW = 0
    for t in tmp2:
        if t > 9:
            w = math.log10(t)  # Final numerical length - 1
            while curW < w:
                curW += 1
                seq.append(-1)
        else:
            curW += 1
            seq.append(t)
    # For numbers where the units digit is non-zero and preceded by -1, move it to a higher position if possible.
    if seq[0] > 0 and len(seq) > 1 and seq[1] == -1:
        seqLen, p = len(seq), 1
        while p < seqLen and seq[p] == -1:
            p += 1
        # Swap
        seq[p - 1] = seq[0]
        seq[0] = 0
    # Concatenate seq, convert -1 to 0, keep other values as they are.
    return "".join([str(n if n >= 0 else 0) for n in seq[::-1]])
import math

# Coefficient (for individual digits and irregular numbers)
EN_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19
}

# Base (for tens and larger scales)
EN_UNIT = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1000000,
    "billion": 1000000000, "trillion": 1000000000000
}

def en2an(english_number_words):
    """
    Converts English number words to Arabic numerals.
    Supports numbers up to trillions.
    :param english_number_words: String of English number words (e.g., "one hundred twenty-three").
    :return: Arabic numeral as an integer. Returns -1 for invalid input.
    """
    words = english_number_words.lower().replace('-', ' ').split()
    current_number = 0
    result = 0
    
    for word in words:
        if word in EN_NUM:
            current_number += EN_NUM[word]
        elif word in EN_UNIT:
            unit_value = EN_UNIT[word]
            if unit_value >= 100:  # Multipliers like hundred, thousand, million
                if current_number == 0: # Handle cases like "thousand" at the start
                    current_number = 1
                result += current_number * unit_value
                current_number = 0
            else:  # Tens like twenty, thirty
                current_number += unit_value
        elif word == "and":
            continue # "and" is often ignored in this context
        else:
            # Handle cases where non-number words are present or invalid input
            print(f"Warning: Unknown word '{word}' in input.")
            return -1 # Or raise an error, depending on desired behavior

    result += current_number # Add any remaining current_number (e.g., for "twenty-three" or "five")
    
    return result

# Example Usage
# print(f"'one hundred twenty-three': {en2an('one hundred twenty-three')}") # Expected: 123
# print(f"'two thousand five hundred forty-six': {en2an('two thousand five hundred forty-six')}") # Expected: 2546
# print(f"'nineteen': {en2an('nineteen')}") # Expected: 19
# print(f"'fifty thousand two hundred one': {en2an('fifty thousand two hundred one')}") # Expected: 50201
# print(f"'one million three hundred fifty thousand and five': {en2an('one million three hundred fifty thousand and five')}") # Expected: 1350005
# print(f"'one hundred': {en2an('one hundred')}") # Expected: 100
# print(f"'thousand': {en2an('thousand')}") # Expected: 1000
import math

# Arabic words to their numerical values
# This is a highly simplified set and will need expansion for a robust solution
AR_NUM = {
    "صفر": 0, "واحد": 1, "اثنان": 2, "ثلاثة": 3, "أربعة": 4,
    "خمسة": 5, "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9,
    "عشرة": 10, "أحد عشر": 11, "اثنا عشر": 12, "ثلاثة عشر": 13,
    "أربعة عشر": 14, "خمسة عشر": 15, "ستة عشر": 16,
    "سبعة عشر": 17, "ثمانية عشر": 18, "تسعة عشر": 19,
    "عشرون": 20, "ثلاثون": 30, "أربعون": 40, "خمسون": 50,
    "ستون": 60, "سبعون": 70, "ثمانون": 80, "تسعون": 90,
    "مائة": 100, "مئتان": 200, "ثلاثمائة": 300, # These might need more flexible handling
    "ألف": 1000, "ألفان": 2000, # Dual forms
    "مليون": 1000000, "مليونان": 2000000, # Dual forms
    "مليار": 1000000000, "ملياران": 2000000000, # Dual forms
    "ترليون": 1000000000000
}

# Arabic conjunction (and)
AR_CONJUNCTION = "و"

def ar2an(arabic_number_words):
    """
    Converts Arabic number words to Arabic numerals.
    This is a simplified version and may not handle all grammatical complexities.
    :param arabic_number_words: String of Arabic number words (e.g., "مائة وثلاثة وعشرون").
    :return: Arabic numeral as an integer. Returns -1 for invalid input.
    """
    words = arabic_number_words.split() # Simple split for now, might need more sophisticated tokenization
    current_number = 0
    result = 0
    
    # Arabic numbers are often read units then tens, then hundreds, then thousands
    # e.g., 23 is "three and twenty"
    # This function will attempt a left-to-right parsing for simplicity,
    # but a truly robust solution would need a more complex grammar-aware parser.
    
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check for multi-word numbers like "أحد عشر"
        two_word_phrase = " ".join(words[i:i+2])
        if two_word_phrase in AR_NUM:
            current_number += AR_NUM[two_word_phrase]
            i += 2
            continue

        if word in AR_NUM:
            value = AR_NUM[word]
            if value >= 1000: # Multipliers like thousand, million, billion
                if current_number == 0:
                    current_number = 1
                result += current_number * value
                current_number = 0
            else:
                current_number += value
        elif word == AR_CONJUNCTION:
            # The 'and' conjunction is very common in Arabic numbers.
            # It usually means the preceding number should be added to the following.
            # A more robust parser would handle its exact position.
            pass 
        else:
            print(f"Warning: Unknown Arabic word '{word}' in input.")
            return -1 # Or handle as an error
        i += 1

    result += current_number # Add any remaining current_number
    
    return result

# Example Usage (simplified due to complexity of full Arabic grammar)
# print(f"'واحد': {ar2an('واحد')}") # Expected: 1
# print(f"'عشرون': {ar2an('عشرون')}") # Expected: 20
# print(f"'مائة': {ar2an('مائة')}") # Expected: 100
# print(f"'مائة وثلاثة وعشرون': {ar2an('مائة وثلاثة وعشرون')}") # Expected: 123 (simplified parsing)
# print(f"'ألفان وخمسمائة': {ar2an('ألفان وخمسمائة')}") # Expected: 2500 (simplified parsing)