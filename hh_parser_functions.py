import pandas as pd
import numpy as np
import pymorphy2
import matplotlib.pyplot as plt
import requests
import json
import string
import time
import random
from forex_python.converter import CurrencyRates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def area_tree_lookup(tree, query):
    for area in tree:
        if query in (area['name'].lower()):
            return area['id']
        elif area['areas']:
            if area_tree_lookup(area['areas'], query):
                return area_tree_lookup(area['areas'], query)

def area_parse(area_name):
    area_name = area_name.lower()
    area_tree = get("https://api.hh.ru/areas")
    area_id = area_tree_lookup(area_tree, area_name)
    if not area_id:
        print('Неправильно введен город/регион.')
    else:
        return int(area_id)

def get(query, timeout=5, max_retries=5, backoff_factor=0.3):
    """ Выполнить GET-запрос

    :param query: запрос с адресом
    :param timeout: максимальное время ожидания ответа от сервера
    :param max_retries: максимальное число повторных запросов
    :param backoff_factor: коэффициент экспоненциального нарастания задержки
    """
    delay = 0
    for i in range(max_retries):
        try:
            response = requests.get(query)
            return response.json()
        except:
            pass
        time.sleep(delay)
        delay = min(delay * backoff_factor, timeout)
        delay += random.random()
    return response

def get_id_page_counts(job_name, area):
    print('Загрузка вакансий по запросу "', job_name, '"...', sep='')
    query_data = {
        "job_name": job_name,
        "area": area
    }
    query = "https://api.hh.ru/vacancies?text={job_name}&area={area}".format(**query_data)
    result = get(query)
    # print(result)
    return (result['found'], result['pages'])

def get_vacancy_ids(job_name, area):
    query_data = {
        "job_name": job_name,
        "page": 0,
        "area": area
    }
    id_count, page_count = get_id_page_counts(job_name, area)
    id_list = []
    while query_data['page']<page_count:
        query = "https://api.hh.ru/vacancies?text={job_name}&page={page}&area={area}".format(**query_data)
        result = get(query)
        query_data["page"] += 1
        for i in range(len(result['items'])):
            id_list.append(result['items'][i]['id'])
    return id_list

def parse_append(raw_data, json_data):
    try:
        if not json_data:
            json_data={
                    'id': [],
                    'name': [],
                    'schedule': [],
                    'employment':[],
                    'experience':[],
                    'salary_min': [],
                    'salary_max': [],
                    'currency': [],
                    'description': []
        }
        json_data['id'].append(raw_data['id'])
        json_data['name'].append(raw_data['name'])
        json_data['schedule'].append(raw_data['schedule']['id'])
        json_data['employment'].append(raw_data['employment']['id'])
        json_data['experience'].append(raw_data['experience']['id'])
        if raw_data['salary']:
            if raw_data['salary']['from'] == None: 
                json_data['salary_min'].append(raw_data['salary']['to'])
            else:
                json_data['salary_min'].append(raw_data['salary']['from'])
            if raw_data['salary']['to'] == None: 
                json_data['salary_max'].append(raw_data['salary']['from'])
            else:
                json_data['salary_max'].append(raw_data['salary']['to'])
            json_data['currency'].append(raw_data['salary']['currency'])
        else:
            json_data['salary_min'].append(None)
            json_data['salary_max'].append(None)
            json_data['currency'].append(None)
        json_data['description'].append(raw_data['description'])
        return json_data
    except:
        print(raw_data)

def ids_to_data(id_list, area):
    job_json = {}
    for i in range(len(id_list)):
        query = "https://api.hh.ru/vacancies/" + id_list[i]
        job_json = parse_append(get(query), job_json)
    return job_json        

def get_job_data(job_name, count=None, area=113):
    id_list = get_vacancy_ids(job_name, area)
    if count:
        return ids_to_data(id_list[:count], area)
    return ids_to_data(id_list, area)

def normalize_currency(job_entry):
    c = CurrencyRates()
    if job_entry.currency == 'USD':
        job_entry.salary *= round(float(c.get_rate('USD', 'RUB'))) 
    if job_entry.currency == 'EUR':
        job_entry.salary *= round(float(c.get_rate('EUR', 'RUB')))
    return job_entry

def normalize_salary(job_df):
    job_df.insert(loc=4, 
                  column='salary', 
                  value=((job_df['salary_min']+job_df['salary_max'])/2))
    job_df = job_df.apply(normalize_currency, axis=1)
    return job_df.drop(columns=['salary_min', 'salary_max', 'currency'])

def vectorize_description(desc_dict, stop_words):
    count_list = []
    for job in desc_dict.keys():
        text = desc_dict[job]
        count_vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,3))
        count_data = count_vectorizer.fit_transform([text])
        words = count_vectorizer.get_feature_names()
        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts+=t.toarray()[0]
        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[:]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        word_counts = pd.DataFrame(words, range(len(words)), columns=['words'])
        word_counts['counts'] = counts
        count_list.append((job, word_counts.head(20)))
    return count_list

def tfidf(desc_dict, stop_words):
    desc_list = [(desc_dict[job], job) for job in desc_dict.keys()]
    morph = pymorphy2.MorphAnalyzer()
    corpus = []
    name_list = []
    deleted_symbols = string.punctuation + '0123456789'
    for desc in desc_list:
        line = desc[0].translate(str.maketrans(deleted_symbols, ' '*len(deleted_symbols)))
        normalized_text = ' '.join(morph.parse(word)[0].normal_form for word in line.split())
        corpus.append(normalized_text)
        name_list.append(desc[1])
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=0.5)
    tfidf_data = tfidf_vectorizer.fit_transform(corpus)
    return tfidf_vectorizer, tfidf_data, name_list

def translate_employment(label):
    if label == 'full':
        return 'Полная занятость'
    elif label == 'part':
        return 'Частичная занятость'
    elif label == 'project':
        return 'Проектная работа'
    elif label == 'probation':
        return 'Стажировка'
    return label
    
def translate_schedule(label):
    if label == 'fullDay':
        return 'Полный день'
    elif label == 'flexible':
        return 'Гибкий график'
    elif label == 'shift':
        return 'Сменный график'
    elif label == 'remote':
        return 'Удаленная работа'
    elif label == 'flyInFlyOut':
        return 'Вахтовый метод'
    return label    

def translate_experience(label):
    if label == 'noExperience':
        return 'Нет опыта'
    elif label == 'between1And3':
        return 'От 1 до 3 лет'
    elif label == 'between3And6':
        return 'От 3 до 6 лет'
    elif label == 'moreThan6':
        return 'Более 6 лет'
    return label

def input_data():
    job_list = []
    query = ''
    query_number = 0
    area = input('Введите регион поиска вакансий:')
    max_samples = int(input('Введите макс. кол-во вакансий по каждому запросу:'))
    while (query != '' or query_number == 0):
        query_number += 1
        query = input('Введите название вакансии ' + str(query_number) + ':')
        if query:
            job_list.append(query)
    return job_list, area, max_samples
            
def download_data(job_list, area, max_samples):
    
    if area:
        area_id = area_parse(area)
    
    job_df_list = {}
    for i in range(len(job_list)):
        job_df_list[job_list[i]] = {}
        
    if area:
        for job in job_list:
            job_df_list[job] = pd.DataFrame(get_job_data(job, count = max_samples, area = area_id))
    else:
        for job in job_list:
            job_df_list[job] = pd.DataFrame(get_job_data(job, count = max_samples))
            
    for job in job_list:
        if job == job_list[0]:
            job_df_full = job_df_list[job].head(max_samples)
            job_df_full.insert(loc=2, column='job_type', value=job)
        else:
            job_df_full = job_df_full.append(job_df_list[job])
            job_df_full['job_type'] = job_df_full['job_type'].fillna(job)
            
    job_df_full = normalize_salary(job_df_full)
    job_df_full['employment'] = job_df_full['employment'].apply(translate_employment)
    job_df_full['schedule'] = job_df_full['schedule'].apply(translate_schedule)
    job_df_full['experience'] = job_df_full['experience'].apply(translate_experience)
    
    with open('stop_words.txt', encoding = 'utf8') as file:
        stop_words = file.readlines()
    stop_words = [i[:-1] for i in stop_words]
    extra_words = [
        'наличие',
        'медицинский',
        'соблюдение',
        'обязанность',
        'высокий',
        'средний',
        'заработный',
        'плата',
        'тыс',
        'руб',
    ]
    stop_words += extra_words
    
    morph = pymorphy2.MorphAnalyzer()
    for job in job_list:
        for word in job.split():
            stop_words.append(morph.parse(word)[0].normal_form)
    #for word in area.split():
    #    stop_words.append(morph.parse(word)[0].normal_form)
            
    desc_dict = job_df_full.groupby('job_type')['description'].apply(','.join).to_dict()
    job_df_full = job_df_full.drop(columns=['description'])
    
    vec, data, job_names = tfidf(desc_dict, stop_words)
    
    tfidf_dict = {}
    for i in range(len(job_list)):
        df = pd.DataFrame(data[i].T.todense(), index=vec.get_feature_names(), columns=["TF-IDF"])
        df = df.sort_values('TF-IDF', ascending=False)
        tfidf_dict[job_names[i]] = df
    
    return job_df_full, tfidf_dict

def draw_employment_pie_chart(job_df_full):
    job_df_class = job_df_full.groupby('employment').size()
    job_df_class.plot(kind='pie', figsize=(6, 6), label="", autopct='%1.0f%%')
    plt.title("Тип занятости")
    plt.show()

def draw_schedule_pie_chart(job_df_full):
    job_df_class = job_df_full.groupby('schedule').size()
    job_df_class.plot(kind='pie', figsize=(6, 6), label="", autopct='%1.0f%%')
    plt.title("График работы")
    plt.show()

def draw_experience_pie_chart(job_df_full):
    job_df_class = job_df_full.groupby('experience').size()
    job_df_class.plot(kind='pie', figsize=(6, 6), label="", autopct='%1.0f%%')
    plt.title("Требуемый опыт работы")
    plt.show()

def draw_employment_histogram(job_df_full):
    x_var = 'job_type'
    groupby_var = 'employment'
    job_df_agg = job_df_full.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [job_df_full[x_var].astype('category').cat.codes.values.tolist() for i, job_df_full in job_df_agg]
    plt.figure(figsize=(10,5))
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, job_df_full[x_var].unique().__len__(), color=colors[:len(vals)], rwidth=0.5, stacked=True, density=False)
    plt.legend({group:col for group, col in zip(np.unique(job_df_full[groupby_var]).tolist(), colors[:len(vals)])})
    plt.xticks(bins[:-1], np.unique(job_df_full[x_var]).tolist(), rotation=20, horizontalalignment='left')
    plt.title("Тип занятости")
    plt.ylabel('Количество вакансий')
    plt.show()
    
def draw_schedule_histogram(job_df_full):
    x_var = 'job_type'
    groupby_var = 'schedule'
    job_df_agg = job_df_full.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [job_df_full[x_var].astype('category').cat.codes.values.tolist() for i, job_df_full in job_df_agg]
    plt.figure(figsize=(10,5))
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, job_df_full[x_var].unique().__len__(), color=colors[:len(vals)], rwidth=0.5, stacked=True, density=False)
    plt.legend({group:col for group, col in zip(np.unique(job_df_full[groupby_var]).tolist(), colors[:len(vals)])})
    plt.xticks(bins[:-1], np.unique(job_df_full[x_var]).tolist(), rotation=20, horizontalalignment='left')
    plt.title("График работы")
    plt.ylabel('Количество вакансий')
    plt.show()

def draw_experience_histogram(job_df_full):
    x_var = 'job_type'
    groupby_var = 'experience'
    job_df_agg = job_df_full.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [job_df_full[x_var].astype('category').cat.codes.values.tolist() for i, job_df_full in job_df_agg]
    plt.figure(figsize=(10,5))
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, job_df_full[x_var].unique().__len__(), color=colors[:len(vals)], rwidth=0.5, stacked=True, density=False)
    plt.legend({group:col for group, col in zip(np.unique(job_df_full[groupby_var]).tolist(), colors[:len(vals)])})
    plt.xticks(bins[:-1], np.unique(job_df_full[x_var]).tolist(), rotation=20, horizontalalignment='left')
    plt.title("Требуемый опыт работы")
    plt.ylabel('Количество вакансий')
    plt.show()

def draw_salary_ratio(job_df_full):
    job_df_class = job_df_full.groupby(job_df_full['salary']>0).size()
    job_df_class.plot(kind='bar',figsize=(6, 5))
    plt.legend(['Количество вакансий'])
    plt.xticks(ticks = [0,1], labels = ['Зарплата не указана', 'Зарплата указана'], rotation = 0)
    plt.title("Соотношение вакансий с указанной заработной платой и без")
    plt.xlabel(' ')
    plt.show()

def draw_salary_histogram(job_df_full):
    x_var = 'job_type'
    job_df_full['has_salary'] = job_df_full['salary'] > 0
    groupby_var = 'has_salary'
    job_df_agg = job_df_full.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [job_df_full[x_var].astype('category').cat.codes.values.tolist() for i, job_df_full in job_df_agg]
    plt.figure(figsize=(10,5))
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, job_df_full[x_var].unique().__len__(), color=colors[:len(vals)], rwidth=0.5, stacked=True, density=False)
    plt.legend(['Зарплата не указана', 'Зарплата указана'])
    plt.xticks(bins[:-1], np.unique(job_df_full[x_var]).tolist(), rotation=20, horizontalalignment='left')
    plt.title("Вакансии, у которых указана заработная плата")
    plt.ylabel('Количество вакансий')
    plt.show()
    

def draw_avg_salary_histogram(job_df_full):
    job_df_class = job_df_full[job_df_full['salary']>0].groupby('job_type').sum()
    job_df_class['salary'] /= job_df_full[job_df_full['salary']>0].groupby('job_type').size()
    job_df_class.plot(kind='barh', y='salary', figsize=(6, 6), label="")
    plt.ylabel(' ')
    plt.legend(['Руб. в месяц'])
    plt.title("Средняя заработная плата")
    plt.show()

def draw_tfidf_histogram(tfidf_dict, job_list):
    remove_repeated_ngrams(tfidf_dict, job_list)
    for job in job_list:
        tfidf_dict[job].head(20).sort_values(by='TF-IDF', ascending=True).plot(kind='barh')
        plt.title(job)
        plt.show()

def remove_repeated_ngrams(tfidf_data, job_list):
    for job in job_list:
        edits_made = True
        shift = 0
        while edits_made:
            edits_made = False
            for ngram in tfidf_data[job][0+shift:22].index:
                if len(ngram.split()) > 1:
                    for word in ngram.split():
                        tfidf_data[job].drop(tfidf_data[job][tfidf_data[job].index == word].index, inplace=True)
                        shift -= 1
                        edits_made = True
            shift += 20
