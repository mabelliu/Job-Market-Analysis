import os
from werkzeug.utils import secure_filename

from flask import Flask

from flask import session, url_for, render_template , request, redirect
from wtforms import SelectField

#from flask.ext.mysql import MySQL

from wtforms.fields.html5 import DateField
from datetime import date

from flask_wtf import FlaskForm

#from flaskext.mysql import MySQL
from flask_wtf import Form
from wtforms.fields.html5 import DateField,IntegerField
from wtforms import SelectField, RadioField, BooleanField,validators
#import geopandas as gpd
import seaborn as sns
import os
import csv
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


from datetime import date

import numpy as np 
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import sent_tokenize,word_tokenize
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.probability import FreqDist
#from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import BigramCollocationFinder
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import networkx as nx
import math 


app = Flask(__name__)


app.secret_key = 'A0Zr98slkjdf984jnflskj_sdkfjhT'
UPLOAD_FOLDER = '/Users/sammi/Downloads/da'

def get_dat_data():
    dat = pd.read_csv('/Users/sammi/Downloads/ba/dat.csv')
    dat = dat[~dat['STATE'].isin(['VI', 'PW', 'GU', 'FM', 'MP'])]
    return dat
#==============================================================================
# 
# def get_homepage_links():
# 	# print(url_for('map'))
# 
# 	return 	[	{"href": url_for('map'), "label":"US H1B Map"},
# 						{"href": url_for('analytics'), "label":"H1B Data Analytics"},
#                         {"href": url_for('analytics1'), "label":"Job and Skill Analytics"},]
#==============================================================================

class AnalyticsForm(FlaskForm):
	attributes = SelectField('Data Attributes', choices=[('EMPLOYER_NAME','Employer Name'),('PREVAILING_WAGE','Salary'),('STATE', 'State'), ('JOB_TITLE', 'Job Title'),('YEAR','Year')])
	attributes1 = SelectField('Data Attributes', choices=[('EMPLOYER_NAME','Employer Name'),('PREVAILING_WAGE','Salary'),('STATE', 'State'), ('JOB_TITLE', 'Job Title'),('YEAR','Year')])


class MyClass(object):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
 #   plt.style.use('seaborn')#looks nicer
#    import geopandas as gpd
#    import seaborn as sns
    def __init__(self, df):
        self.df = df

    def sort_img(self,other):
        col=other
        img = self.df.groupby(col).size().nlargest(53).sort_values(ascending=False).plot(kind='bar',figsize=(20,4),color="orange",title="Applications per {}".format(col))
        img.set_xlabel(col)
        img.set_ylabel("Applications")
        img.grid(True)
        fig1=img.get_figure()
        return fig1.savefig("static/figg1%s.png"%col)
        # return fig1.savefig("static/figg1")

    # def dist_ply(self):
    #     col="STATE"
    #     # col=input("variables: ")
    #     img2=sns.distplot(self.df.groupby(col).size().nlargest(53).sort_values(ascending=False))
    #     fig2=img2.get_figure()
    #     fig2.savefig("static/fig2.png")
    def pairplot(self):   
        img5=sns.pairplot(x_vars=['STATE'], y_vars=['PREVAILING_WAGE'], data=self.df, hue="YEAR", size=12)
      
        return img5.savefig("static/figg5.png")
    def binary_img(self,other,other1):
        col1=other
        col2=other1
        # col1=input("variables1: ")
        # col2=input("variables2: ")
        col = self.df.groupby([col1,col2])
        STATE_YEAR_plt=col.size().nlargest(371).unstack().plot(kind='bar',figsize=(12,12),title="Application each {} for {}".format(col1,col2))
        STATE_YEAR_plt.set_xlabel("Applications")
        STATE_YEAR_plt.set_ylabel(col)
        STATE_YEAR_plt.grid(True)
        fig3=STATE_YEAR_plt.get_figure()

        return fig3.savefig("static/figg3%s%s.png"%(col1,col2))
        # return fig3.savefig("static/figg3.png")

    def multiple_img(self,other,other1):
        col1=other1
        col11=col1
        col2=other
        col22=col2

# col1=input("variable1: ")
# col2=input("variable2: ")
        year_state = self.df.groupby([col1, col2]).size().unstack()
        COL_NUM = 3
        ROW_NUM = len(self.df[col2].unique())//3+1
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(ROW_NUM, COL_NUM, figsize=(20,30))

        for i, (col2,col1) in enumerate(year_state.items()): #i is used to position
            ax = axes[int(i/COL_NUM), i%COL_NUM]
            col1= col1.sort_values(ascending=False)[:6] 
            col1.plot(kind='barh', ax=ax)
            ax.set_title(col2)

        plt.tight_layout()

        return plt.savefig("static/figg4%s%s.png"%(col11, col22))
        # return plt.savefig("static/figg4.png")

def average_data():
    # col=input('variables:')
    # col1=input('datset column:')
    # col2=input('specific:')
    dat=get_dat_data()
    col='STATE'
    col1='JOB_TITLE'
    col2='PROGRAMMER ANALYST'
    if col1 and col2:
        narrow_dat=dat[dat[col1]==col2]
    else:
        narrow_dat=dat
    mean_dat=pd.DataFrame(narrow_dat.groupby([col]).mean()['PREVAILING_WAGE'])
    count_dat=pd.DataFrame(narrow_dat.groupby([col]).count()['EMPLOYER_NAME'])
    img_dat=count_dat.join(mean_dat)
    img_dat=img_dat.rename(columns = {'EMPLOYER_NAME':'APPLICATIONS'})
    return img_dat

    
    
#analyze indeed data 

class TextAnalysis:
#change the directory to the one that includes indeed-scrap-output1 file
#pre define junk_words that we do not like to count frequency and programming skills
    junk_words=['contact','us','about','email','address','first','name','last', 'job', 'all', 'search','terms','conditions','our', 'more', 'sign','jobs','ny','york','united'
                ,'states','san','francisco','privacy','term','san','years']
    junk_words.extend(stopwords.words())
    # r subject to change to the capitalized one
    programming_skills=['excel','vba','python','r','matlab','c#','c++','saas','stata','sql','mysql','php','html','java','javascript',
    'ios','perl']
    


# initilize the class object feature by reading corpus from specific job title folder including a collection of job postings files
    def __init__(self,value,junk=junk_words,p=programming_skills):
        self.value=value
        self.junk_words=junk
        self.programming=p
        root= os.getcwd() + '/output/' + self.value
        files=self.value+'.*'
        text =PlaintextCorpusReader(root,files)
        self.corpus=text


# create the bag of words with most 20 frequent terms   
    def bag_of_words(self,toprank=20):
        tokenizer=RegexpTokenizer(r'\w+')
        newtext=self.corpus.raw()
        tokens=tokenizer.tokenize(newtext)
        p_stemmer=PorterStemmer() 
        Output=[p_stemmer.stem(word.lower()) for word in tokens if word.lower() not in self.junk_words and word.isalpha()]
        word_freq=FreqDist(Output).most_common(toprank)
        words=[i[0] for i in word_freq]
        self.words=words

# create the bag of bigram with most 20 frequent terms   
    def bag_of_bigrams(self,toprank=20):
        tokenizer=RegexpTokenizer(r'\w+')
        newtext=self.corpus.raw()
        tokens=tokenizer.tokenize(newtext)        
        words=[word.lower()for word in tokens if word.lower() not in self.junk_words and word.isalpha()]
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(words)
        scored = finder.score_ngrams(bigram_measures.raw_freq)
        freq=sorted(scored,key=lambda scored:scored[1],reverse=True)[:toprank]
    #        vectorizer=CountVectorizer(ngram_range=(1,2))
    #        bag_of_words=vectorizer.fit_transform(words)
    #        freq=sum(bag_of_words).toarray()[0]
            #df=pd.DataFrame(freq,index=bag_of_words.get_feature_name(),columns=['freqency'])
            #bigram=df.sort_values(by=['frequency'])[:toprank].index.values
            #self.bigram=bigram
        bigram=[ rank[0] for rank in freq]
        self.bigram=bigram


  
            
# create programming skills dictionary in terms of frequncy in two ways, self.skills for further analysis, and self.skills_fq for representation  
    def programming_skill(self):
 #       with open(os.getcwd()+'\\indeed-scrap-output-full\\output\\'+self.value,'r') as f:
  #          data=f.read().replace('\n','')
#       import operator
        newtext=self.corpus.raw()
        tokenizer=RegexpTokenizer(r'\w+')
        tokens=tokenizer.tokenize(newtext)
        words=[word.lower()for word in tokens if word.lower() not in self.junk_words and word.isalpha()]
        skill_dict=dict(zip(self.programming,[0]*len(self.programming)))
        for i in words:
            if i in skill_dict.keys():
                skill_dict[i]+=1
            else:
                pass        
      #  skills=sorted(skill_dict.items(),key=operator.itemgetter(1),reverse=True)
        total= sum(skill_dict.values())
        skills={key.capitalize():value/total for key,value in skill_dict.items()}     
        skills_freq={key.capitalize():value/total for key,value in skill_dict.items() if value is not 0} 
        pd.options.display.float_format='{:.2%}'.format  
        skills_freq=pd.Series(skills_freq).sort_values(ascending=False)
        self.skills=skills
        self.skills_fq=skills_freq

       
        

    
# draw the wordcloud of programming skills
    def draw_wordcloud(self,max_words=20):
        skills=dict(self.skills_fq)
        #skills_new={key.capitalize():value for key,value in skills.items() if value is not 0}   

        #        newtext=self.corpus.raw()
        #        tokenizer=RegexpTokenizer(r'\w+')
        #        tokens =tokenizer.tokenize(newtext)
        # words=[word.lower()for word in tokens if word.lower() not in self.junk_words and word.isalpha()]
        wordcloud =WordCloud(background_color='white',width=800,height=600,max_words=max_words).fit_words(skills)
        plt.imshow(wordcloud)
        plt.axis('off')
        return plt.savefig('static/wordcloud%s.png'%self.value)
        
        # plt.savefig('templates/wordcloud.png')
        # plt.show()
        
        
 #draw the wordcloud for the whole corpus      
    def draw_bigram(self,max_words=20):
        #remove short wordse short words 
        #    bigram=dict(self.bigram)

        # bigram_new={key:value for key,value in skills.items() if value is not 0}   

        #        newtext=self.corpus.raw()
        #        tokenizer=RegexpTokenizer(r'\w+')
        #        tokens =tokenizer.tokenize(newtext)
        # words=[word.lower()for word in tokens if word.lower() not in self.junk_words and word.isalpha()]
        wordcloud =WordCloud(stopwords=self.junk_words,background_color='white',width=1200,height=1000,max_words=max_words).generate(self.corpus.raw())
        plt.imshow(wordcloud)
        plt.axis('off')
        
        return plt.savefig('static/bigram%s.png'%self.value)

#-------------------------------
class Job_Title_Form(FlaskForm):
    job_title_list=[
     ('Business Analyst', 'Business Analyst'),
     ('Business Intelligence Engineer', 'Business Intelligence Engineer'),
     ('Consultant', 'Consultant'),
     ('Data Analyst', 'Data Analyst'),
     ('Data Engineer', 'Data Engineer'),
     ('Data Scientist', 'Data Scientist'),
     ('Global Markets Analyst', 'Global Markets Analyst'),
     ('Investment Banking Analyst', 'Investment Banking Analyst'),
     ('Java Developer', 'Java Developer'),
     ('Media Analyst', 'Media Analyst'),
     ('Operations Analyst', 'Operations Analyst'),
     ('Portfolio Analyst', 'Portfolio Analyst'),
     ('Pricing Analyst', 'Pricing Analyst'),
     ('Program Manager', 'Program Manager'),
     ('Project Manager', 'Project Manager'),
     ('Quantitative Analyst', 'Quantitative Analyst'),
     ('Quantitative Trader', 'Quantitative Trader'),
     ('Risk Management', 'Risk Management'),
     ('Strategic Sourcing Manager', 'Strategic Sourcing Manager')]

    job_title = SelectField('Select Job Title', choices=job_title_list, default=job_title_list[0][0])
    
    


#create Network class to draw features between two TextAnalysis objects 
class Network(TextAnalysis):
    def __init__(self,job1):
        self.job=job1
        self.startnode=TextAnalysis(job1)
   #     self.endnode=TextAnalysis(job2)
        self.startnode.programming_skill()
        self.value=self.startnode.value
        self.skills_fq=self.startnode.skills_fq
        self.skills=self.startnode.skills
        self.corpus=self.startnode.corpus
   #     self.endnode.programming_skill()
  #      self.edge2=self.endnode.skills_fq
        job_title=['Business Analyst','Business Intelligence Engineer','Consultant','Data Analyst',
                   'Data Engineer','Data Scientist','Global Markets Analyst','Investment Banking Analyst',
                   'Java Developer','Media Analyst','Operations Analyst','Portfolio Analyst',
                   'Pricing Analyst','Program Manager','Project Manager','Quantitative Analyst','Quantitative Trader',
                   'Risk Management','Strategic Sourcing Manager']
        self.job_title=job_title
  
#==============================================================================
#     def edge_identify(self,job2):
#        
#        endnode=TextAnalysis(job2)
#        endnode.programming_skill()
#        if len(set(self.edge1[:5].index) & set(endnode.skills_fq[:5].index)) >=4:
#            return True
#    #         self
#        else:
#            return False
#==============================================================================

#compute cosine similarity between two job title programming skills dictionary
    def edge_identify(self,job2,threshold=0.8):
        
        endnode=TextAnalysis(job2)
        endnode.programming_skill()
        dic1=self.skills
        dic2=endnode.skills
        numerator = 0
        de1 = 0
        for key1,count1 in dic1.items():
            numerator += count1*dic2.get(key1,0.0)
            de1 += count1*count1
        de2 = 0
        for count2 in dic2.values():
            de2 += count2*count2
        result= numerator/math.sqrt(de1*de2) 
        return result
#==============================================================================
#         if result >=threshold:
#             return True,result
#         else:
#             return False,result
#==============================================================================


 # compute the cosine similarity between the chosen job title and all the other 18 jobs       
    def job_connection(self,threshold=0.8):
        job_con=list()
        for i in self.job_title:
            if i != self.job:          
               job_con.append((i,self.edge_identify(i,threshold)))
            else:
                pass     
        self.job_con=job_con    



#draw the network of job similarity with chosen job type as the center        
    def draw_network(self,threshold=0.8):
        job_network=nx.Graph()
        self.job_connection(threshold)
        job_con=self.job_con
  #      ebold=list()
  #      edash=list()
        for i in range(len(job_con)):
            job_network.add_edge(self.job,job_con[i][0],sim=round(job_con[i][1],2))
               # bold.append((self.job,job_con[i][0],job_con[i][1]))
        elarge=[(u,v) for (u,v,d) in job_network.edges(data=True) if d['sim'] >threshold]
        esmall=[(u,v) for (u,v,d) in job_network.edges(data=True) if d['sim'] <=threshold]

        plt.figure(1,figsize=(12,12)) 
        pos=nx.spring_layout(job_network)   
        nx.draw_networkx_nodes(job_network,pos,
                           node_color='r',
                           node_size=800,
                           alpha=0.6) 
        nx.draw_networkx_edges(job_network,pos,edgelist=elarge,edge_color='b',
                    width=6)

        nx.draw_networkx_edges(job_network,pos,edgelist=esmall,
                    width=6,alpha=0.5,edge_color='b',style='dashed')
        node_name={}
        for node in job_network.nodes():
            node_name[node]=str(node)
        nx.draw_networkx_labels(job_network,pos,node_name,font_size=10,font_family='sans-serif',font_weight='bold')  
        nx.draw_networkx_edge_labels(job_network,pos,font_size=10)


        plt.axis('off')
        plt.savefig("job_network.png")
        return plt.savefig('static/network%s.png'%self.job)


#---------------------------


#build up our flask app to show all the above features 

@app.route("/")
def home():
	session["data_loaded"] = True
	return render_template('home.html')
#, links=get_homepage_links()

@app.route("/map")
def map():
# def mapping():
#    import plotly.plotly as py
#    import pandas as pd
    import plotly 
    img_pa=average_data()
    plotly.tools.set_credentials_file(username='sammixue', api_key='g0dDah0MbtDZR6475v4N')


    scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
                [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


    img_pa['text'] = img_pa.index + '<br>' +\
        'Applications '+img_pa['APPLICATIONS'].astype(str)+' Avg Salary '+img_pa['PREVAILING_WAGE'].astype(str)


    data1 = [ dict(
            type='choropleth',
            colorscale = scl,
            autocolorscale = False,
            locations = img_pa.index,
            z = img_pa['APPLICATIONS'],
            locationmode = 'USA-states',
            text = img_pa['text'],
            marker = dict(
                line = dict (
                    color = 'rgb(255,255,255)',
                    width = 2
                ) ),
            colorbar = dict(
                title = "Applications")
            ) ]
    layout = dict(
            title = 'Applications per State',
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor = 'rgb(255, 255, 255)'),
                 )
    fig = dict( data=data1, layout=layout )
    plotly.offline.plot(fig, filename='templates/file.html')
    return render_template('map.html', mapfile = 'file.html')

@app.route('/analytics/',methods=['GET','POST'])
def analytics():
    form = AnalyticsForm()
    if form.validate_on_submit():
        dat=get_dat_data()
        a=MyClass(dat)
        other=request.form.get('attributes')
        print(other)
        other1=request.form.get('attributes1')
        print(other1)
        a.sort_img(other)
        # a.pairplot()


        a.binary_img(other,other1)

        a.multiple_img(other,other1)
        newname_1 = "figg1%s.png"%other
        newname_2 = "figg5.png"
        newname_3 = "figg3%s%s.png"%(other,other1)
        newname_4 = "figg4%s%s.png"%(other1,other)


  

        return render_template('analyticsoutput.html', newname_1 = newname_1, newname_2=newname_2,newname_3=newname_3, newname_4=newname_4,other = other, other1=other1)

    return render_template('analyticsparams.html', form=form)
@app.route('/analytics1/',methods=['GET','POST'])
def analytics1():
    form = Job_Title_Form()
    if form.validate_on_submit():
        j_title = request.form.get('job_title')
        k=Network(j_title)
        k.draw_network(threshold=0.85)
        # j_title = 'Consultant'
        k.draw_wordcloud()  
        new_wc_name = 'wordcloud%s.png'%k.value
        k.draw_bigram(100)
        new_br_name = 'bigram%s.png'%k.value

        
        # k.programming_skill() 
        # l.skills_fq
        new_nk_name = 'network%s.png'%k.job


        return render_template('analyticsoutput1.html',new_wc_name=new_wc_name,new_br_name = new_br_name, new_nk_name=new_nk_name, job_to_ht = j_title) 
        #return render_template('analyticsoutput.html', job_to_ht = j_title)



    return render_template('jobparams.html', form=form)
# def analytics1():
#     form = Job_Title_Form()
#     if form.validate_on_submit():
#         j_title = request.form.get('job_title')
#         k=Network(j_title)
#         k.draw_network(threshold=0.85)
#         # j_title = 'Consultant'
#         k.draw_wordcloud()  
#         new_wc_name = 'wordcloud%s.png'%k.value
#         k.draw_bigram(100)
#         new_br_name = 'bigram%s.png'%k.value

        
#         # k.programming_skill() 
#         # l.skills_fq
#         new_nk_name = 'network%s.png'%k.job


#         return render_template('analyticsoutput1.html',new_wc_name=new_wc_name,new_br_name = new_br_name, new_nk_name=new_nk_name, job_to_ht = j_title) 
#         #return render_template('analyticsoutput.html', job_to_ht = j_title)



#     return render_template('jobparams.html', form=form)


if __name__ == '__main__':
	app.run(debug = True)