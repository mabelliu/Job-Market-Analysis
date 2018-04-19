#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import argv
import requests
import math
from bs4 import BeautifulSoup
import os
import re
from time import sleep
import random


# scrape process flow:
#1.search the job w/o location
#2.get the location li: location+job count + job_location_url
#3.jump to job_location_url
#4.for each div class "row result" find href
#    results_col ->3 blocks -> row result/ organicJob -> jobtitle & company
#    for each div row result/ organicJob find the job_detail_url:
#        collect all li values
#        handle expections
#    flip page job_location_url + &start=10x(page_no - 1)


# initialize the scrape
# input: job_keyword
# output: a set of top locations for the job w/. url by location
def job_search_setup(job_keyword):

    print('job title:' + job_keyword.replace(' ', '+'))

    job_url = "https://www.indeed.com/jobs?q={0}&l=".format(job_keyword.replace(' ', '+'))
    base_txt = "https://www.indeed.com"
    location_set = list()
    results_page = BeautifulSoup(requests.get(job_url).content, 'lxml')
#    the keyword for top location block is onmousedown = "rbptk('rb', 'loc', ....)
    loc_tags_all = results_page.find_all('li', onmousedown = re.compile("^rbptk\('rb', 'loc', ") )

    for tag in loc_tags_all:
#        create a list with 3 elements to store one location entry
        location_info = list()
#        use regular expression to remove comma and brackets
        p = re.compile('\(|\)|,')
        location_info.append(str(tag.a.contents[0]))
        location_info.append(float(p.sub('', tag.contents[-1].strip())))
        location_info.append( base_txt + tag.a.get('href'))
        location_set.append(location_info)
    return location_set


# start search by location
def get_result_by_location(location_set):
# this is a rough estimate of number of job posting to scrape
# as the number will always be a multiplier of ten in each location
    print('total number of job to scrape:', int(sum([loc[1] for loc in location_set]) * scope))
    
    all_jobs_in_keywords = list()
    for loc in location_set:
#        make_folder(base_directory + '\\' + loc[0] )
        print('current location:', loc[0], 'with job count:', str(loc[1]))
#        use extend method to keep the flatten list structure
#        number of rows will be number of job postings
#        last element of each row [:][-1] is a list of job requirements 
        all_jobs_in_keywords.extend(get_job_detail(*loc))
        
# add an element of job keyword for each record
    for entry in all_jobs_in_keywords:
        entry.insert(0, job_keyword)
    
    return all_jobs_in_keywords

def get_job_detail(location_txt, job_quantity, url):
#    location_txt, job_quantity, url = loc
    scrape_stop_count = math.ceil(job_quantity * scope/10)
    all_jobs_in_location = list()
#    search job start with page = 1
    for i in range(scrape_stop_count):
        url_suffix = '&start={0}'.format(i*10)
        print('current page:' + url_suffix)
        
#        get all job posting from the page and append to the list
        all_jobs_in_location.extend(get_all_jobs_from_page(url+url_suffix))
#    add one element of location to the records
    for entry in all_jobs_in_location:
        entry.insert(0, location_txt)
     
    return all_jobs_in_location

def get_all_jobs_from_page(url):
    results_page = BeautifulSoup(requests.get(url).content, 'lxml')
    #"class_":"  row  result", 
    loc_tags_all = results_page.find_all( 'div', {"data-tn-component":"organicJob"})
    job_posting = list()
    all_job_in_page = list()
    for loc_tag in loc_tags_all:
        print(loc_tag.h2.a.get('title'), loc_tag.span.text.strip())
#        print(loc_tag.h2.a.get('href'))
        job_posting = list()
        job_posting.append(loc_tag.h2.a.get('title'))
        job_posting.append(loc_tag.span.text.strip())
#        sleep(random.random())
#        retrive the job details as list and save the list as last element of job posting list
        job_posting.append(retrieve_posting_detail_page(loc_tag.h2.a.get('href')))
        
        all_job_in_page.append(job_posting)
    return all_job_in_page
    
def retrieve_posting_detail_page(url):
#    url = loc_tag.h2.a.get('href')
    base_txt = "https://www.indeed.com"
    li_list = list()
    try:
#        add timeout otherwise some website will keep the program hanging forever
        posting_page = BeautifulSoup(requests.get(base_txt + url, timeout=5).content, 'lxml')
        li_set = posting_page.find_all('li')
#        make sure to not add blanks and li with specific type of fathers
        for x in li_set:
            if bool(x.text.strip()) & common_exclusion(x):
                li_list.append(x.text.strip())
#        li_list = [x.text.strip() for x in li_set]
    except Exception as e:
        print('failed to load job posting page:', str(e) )
    return li_list

# specifically call out some type of ui 
# for example, the embeded linkedin form comes with lot of noisy bullet points.
def common_exclusion(x):
    if x.parent.parent.get('id') == 'eeoc_fields':
        return False
    else:
        return True

# create folder utility
def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# save the final list(all_jobs_in_keyword) to individual files
def save_to_txt(scrape_result):
    base_directory = r'.\output'
    make_folder(base_directory+ '\\' + job_keyword)
    file_name_base = base_directory + '\\' + job_keyword+ '\\' + job_keyword
    i = 0
    for entry in scrape_result:
        i += 1
        thefile = open('{0}.{1}'.format(file_name_base, i), 'w', encoding='utf-8')
        for item in entry:
            if isinstance(item, str):
                thefile.write("%s\n" % item)
            else:
                if isinstance(item, list):
                    for li in item:
                        thefile.write("%s\n" % li)
        thefile.close()
        
#class job_scrapper:
#    
#    def __init__(self, job_keyword, **kwargs):
#        self.job_keyword = job_keyword
#        # pct of jobs need to be scraped
#        self.scope = kwargs.get('scope', 0.05)
#    
#    # collect the job location statistics for setting up formal job searches
#    def job_search_setup(self):
        
if __name__ == '__main__':

    base_directory = r'.\output'
    job_keyword = argv[1]
#    job_keyword = "Project Manager"
    job_url = "https://www.indeed.com/jobs?q={0}&l=".format(job_keyword.replace(' ', '+'))
    scope = 0.05
    
    location_set = job_search_setup(job_keyword)
    scrape_result = get_result_by_location(location_set)
    save_to_txt(scrape_result)