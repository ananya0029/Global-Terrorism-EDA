import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit
import plotly.express as px
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from PIL import Image
import streamlit as st
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import time

st.title("Exploratory Data Analysis: Terrorism")

st.subheader("Author - Ananya Gupta")

st.markdown('Terrorism today is a very much concerning issue. It is no longer specific to a particular country . Even the most powerful economies are finding ways to tackle increasing terrorism. ')
st.markdown('Lets deepdive to analyze past trends of  terrorism by drawing insights from the vast database of ** Global Terrorism from1970-2017** provided to us by **Kaggle**')

image=Image.open('terrorglobal.jpg')
st.image(image,use_column_width=True)
st.write('Image credit: https://www.express.co.uk/travel/articles/1207078/holidays-terror-global-terrorism-index-2019-terrorist-attack-latest')

data_load_state = st.text('Loading Data...')
my_bar = st.progress(0)
for percent_complete in range(100):
     time.sleep(0.1)
     my_bar.progress(percent_complete + 1)
@st.cache
def load_data():
      data=pd.read_csv('globalterrorism.csv',encoding='latin1')#,nrows=5000)
      return(data)


data=load_data()
data_load_state.text('loading data done!')
           

data.rename(columns={'iyear':'Year','imonth':'month','iday':'day','extended':'duration_of_incident','country':'country_code',
                     'country_txt':'country','region':'region_code','region_txt':'region','attacktype1_txt':'Attacktype',
                     'target1':'Target','nkill':'Killed','nwound':'Wounded','gname':'Group',
                     'targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)  
                     

data=data[['Year','month','day','duration_of_incident','country','region','latitude','longitude','Attacktype','Target',
           'Killed','Wounded','Group','summary','Target_type','Weapon_type','Motive']]

data['Casualities']=data['Killed']+data['Wounded']


st.subheader('Most used words in Motive behind attacks')
mask = np.array(Image.open("t.jpg"))
def show_wordcloud(data1):
    stopwords = set(STOPWORDS)
    words_except_stopwords = nltk.FreqDist(w for w in data1 if w not in stopwords) 
    wordcloud = WordCloud(
        background_color='black',
        colormap="summer_r",
        stopwords=stopwords,
        max_words=350,
        mask=mask,
        #max_font_size=40, 
        scale=3,
        contour_width=2, contour_color='black').generate(" ".join(words_except_stopwords))

    fig1 = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    
    plt.imshow(wordcloud, interpolation="bilinear")
    st.pyplot(fig1)

motive=data['Motive'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words=nltk.tokenize.word_tokenize(motive)
show_wordcloud(words)


lats=list(data['latitude'])
lon=list(data['longitude'])

st.markdown(' Words like ** al, non, America, anti, government ** are the most frequently used words in describing the motive behind terror attacks')
st.subheader('High terror zones on world map')

fig=plt.figure(figsize=(12,9))

m = Basemap(projection='mill')
m.drawcoastlines()
m.drawcountries(linewidth=2,color='darkred')
m.drawmeridians(np.arange(0,360,30),labels=[0,0,0,1])
m.drawparallels(np.arange(-90,90,30),labels=[True,False,False,False])
m.etopo()
target_lats,target_lon=m(lon,lats)
m.plot(target_lats,target_lon,'go',markersize=1.5,color = 'r',alpha=0.4)
plt.show()
st.pyplot(fig)

st.markdown("**Terrorist activities were observed to be very geometrically focussed mainly at south east Asia, middle east Asia and Africa**")

if st.checkbox('show Terrorism data'):
       st.subheader('Terrorism dataset(1970-2017)')
       st.write(data)


st.subheader('Countries with high Terror Attacks(1970-2017)')
fig,ax=plt.subplots(figsize=(8,8))
sns.barplot(data['country'].value_counts()[:15].index,data['country'].value_counts()[:15].values,palette='viridis')
plt.xticks(rotation=90)
st.pyplot(fig)
st.markdown("Iraq recorded the highest terror attacks ranging nearly 25000")

st.subheader('Top 10 Countries with highest Casualities')
fig,ax=plt.subplots(figsize=(8,10))
x=data.groupby('country')['Casualities'].sum().sort_values().tail(10)
y = data.groupby('country')['Wounded'].sum().sort_values().tail(10)
z = data.groupby('country')['Killed'].sum().sort_values().tail(10)
x.plot(kind='barh',stacked=True,figsize=(8, 8), color='crimson', zorder=1, width=0.75,legend=True)
y.plot(kind='barh',stacked=True,figsize=(8, 8), color='lime', zorder=1, width=0.75,legend=True)
z.plot(kind='barh',stacked=True,figsize=(8, 8), color='green', zorder=1, width=0.75,legend=True)
st.pyplot(fig)

st.markdown('**Iraq** has encountered most terror attacks and has highest casuality rate.')
st.markdown('Also it can be observed that the **countries with dense popultion** are the top targeted countries')

st.subheader('Regions with high Terror Attacks(1970-2017)')
fig,ax=plt.subplots(figsize=(8,8))
x=data['region'].value_counts()
x.plot(kind='bar',stacked=True,figsize=(8, 8), color='maroon', zorder=1, width=0.65,legend=True)
st.pyplot(fig)

z=data.groupby('region')['Casualities'].mean().sort_values(ascending=False)
st.table(z)

st.markdown('**Damn ! Can you see that East Asia with only few terror attacks has the highest casualities per attack suggesting that the attacks took place in densely populated areas**')

st.subheader("Years with high terrorist activities")
fig1,ax=plt.subplots(figsize=(15,8))
sns.countplot('Year',data=data,palette='rocket',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
st.pyplot(fig1)
st.markdown("Terror activities started rising after 2011 and has the highest peak with nearly **17000 ** terror attacks in 2014 ")


st.subheader('Years with highest Casualities')
fig1,ax=plt.subplots(figsize=(8,8))
x=data.groupby('Year')['Casualities'].sum().sort_values().tail(25)
y = data.groupby('Year')['Wounded'].sum().sort_values().tail(25)
z = data.groupby('Year')['Killed'].sum().sort_values().tail(25)
x.plot(kind='bar',stacked=True,figsize=(8, 8), color='maroon', zorder=1, width=0.45,legend=True)
y.plot(kind='bar',stacked=True,figsize=(8, 8), color='turquoise', zorder=1, width=0.45,legend=True)
z.plot(kind='bar',stacked=True,figsize=(8, 8), color='tomato', zorder=1, width=0.45,legend=True)
st.pyplot(fig1)
st.markdown("**Do you know?** according to global terrorism data, an average of  **{0} attacks happens every year**".format(int(data['Year'].value_counts().mean())))

top_ten_terror_groups=data[data['Group'].isin(data['Group'].value_counts()[1:11].index)]
st.subheader('Most prominent Terrorist groups')
fig,ax=plt.subplots(figsize=(12,10))
data['Group'].value_counts()[0:10].plot.pie(autopct="%.1f%%",startangle = 0)
st.pyplot(fig)

fig,ax=plt.subplots(figsize=(12,10))
group=pd.crosstab(top_ten_terror_groups['Year'],top_ten_terror_groups['Group'])
group.plot(color=sns.color_palette('Paired',10))
fig =plt.gcf()
fig.set_size_inches(10,6)
plt.show()
st.pyplot(fig)
st.markdown('** Taliban and ISIL were the most active terrorist group and the have the highest share in terror attacks after 2010**')


st.subheader('Main targets of Terrorists')
fig=plt.figure(figsize=(12,9))
x=data['Target_type'].value_counts()[:13].plot.pie(autopct="%.1f%%",startangle = 70)
centre_circle = plt.Circle((0,0),0.45,color='black', fc='white',linewidth=1.)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show() 
st.pyplot(fig)
st.markdown('Common Citizens & property, Military , Police were the main targets of terrorists')

st.subheader("Most used weapons by terrorist")

fig,ax=plt.subplots(figsize=(12,6))
group=pd.crosstab(data['Year'],data['Weapon_type'])
group.plot(color=sns.color_palette('Paired',10))
fig =plt.gcf()
fig.set_size_inches(10,6)
plt.show()
st.pyplot(fig)

st.markdown("The use of explosives and firearms is prominent in all the years but an increase can be seen after 2004 that is because of increase in Terrorist activities after 2004")

st.subheader("Duration of Attacks")
fig1,ax=plt.subplots(figsize=(12,8))
sns.countplot('duration_of_incident',data=data,palette='rocket',edgecolor=sns.color_palette('dark',7))
ax.set_xticklabels( ('Less than 24 hours', 'More than 24 hours') )
st.pyplot(fig1)

st.write('Only Few attacks lasted for more than 24 hours')


st.markdown('**I hope the visualization was insightfull.**')

st.markdown("**Thank You for visiting !! **")

st.balloons()









