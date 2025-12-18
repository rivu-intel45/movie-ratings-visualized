# -*- coding: utf-8 -*-
"""
Goal:
Your goal is to complete the tasks below based off the 538 article and see if you reach a similar conclusion. You will need to use your pandas and visualization skills to determine if Fandango's ratings in 2015 had a bias towards rating movies better to sell more tickets.

Part One: Understanding the Background and Data
TASK: Read this article: Be Suspicious Of Online Movie Ratings, Especially Fandango’s

TASK: Import any libraries you think you will use:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""TASK: Run the cell below to read in the fandango_scrape.csv file"""

fandango= pd.read_csv('fandango_scrape.csv')

fandango

"""TASK: Explore the DataFrame Properties and Head."""

fandango.head()

fandango.info()

fandango.describe()

"""TASK: Let's explore the relationship between popularity of a film and its rating. Create a scatterplot showing the relationship between rating and votes. Feel free to edit visual styling to your preference."""

plt.figure(figsize =(10,4), dpi=100)
sns.scatterplot(data=fandango, x='RATING', y='VOTES')

"""TASK: Calculate the correlation between the columns:"""

fandango.drop('FILM',axis=1).corr()

"""TASK: Assuming that every row in the FILM title column has the same format:

Film Title Name (Year)
Create a new column that is able to strip the year from the title strings and set this new column as YEAR
"""

fandango['YEAR']=fandango['FILM'].apply(lambda title:title.split('(')[-1])

fandango['YEAR']

"""TASK: How many movies are in the Fandango DataFrame per year?"""

fandango.value_counts()

fandango['YEAR'].value_counts()

"""TASK: Visualize the count of movies per year with a plot:"""

sns.countplot(data=fandango, x='YEAR')

"""TASK: What are the 10 movies with the highest number of votes?"""

fandango.sort_values('VOTES', ascending =False)[:10]

"""TASK: How many movies have zero votes?"""

(fandango['VOTES']==0).sum()

"""TASK: Create DataFrame of only reviewed films by removing any films that have zero votes.


"""

fan_review=fandango[fandango['VOTES']>0]

fan_review

"""As noted in the article, due to HTML and star rating displays, the true user rating may be slightly different than the rating shown to a user. Let's visualize this difference in distributions.

TASK: Create a KDE plot (or multiple kdeplots) that displays the distribution of ratings that are displayed (STARS) versus what the true rating was from votes (RATING). Clip the KDEs to 0-5.
"""

sns.kdeplot(data=fan_review,x='RATING',clip=(0,5),fill=True,label='True Rating')
sns.kdeplot(data=fan_review,x='STARS',clip=(0,5),fill=True,label='Stars displayed')
plt.legend(loc=(1.05,0.5))

"""TASK: Let's now actually quantify this discrepancy. Create a new column of the different between STARS displayed versus true RATING. Calculate this difference with STARS-RATING and round these differences to the nearest decimal point.


"""

fan_review = fan_review.copy()
fan_review['STARDIFF']=fan_review['STARS']-fan_review['RATING']
fan_review['STARDIFF']=fan_review['STARDIFF'].round(2)

fan_review

plt.figure(figsize=(12,4),dpi=100)
sns.countplot(data=fan_review,x='STARDIFF', palette='magma')

"""TASK: We can see from the plot that one movie was displaying over a 1 star difference than its true rating! What movie had this close to 1 star differential?"""

fan_review[fan_review['STARDIFF']==1]

"""TASK: Read in the "all_sites_scores.csv" file by running the cell below"""

allsites=pd.read_csv('all_sites_scores.csv')

"""TASK: Explore the DataFrame columns, info, description."""

allsites.head()

allsites.info()

allsites.describe()

allsites

"""TASK: Create a scatterplot exploring the relationship between RT Critic reviews and RT User reviews."""

plt.figure(figsize=(10,5),dpi=200)
sns.scatterplot(data=allsites,x='RottenTomatoes',y='RottenTomatoes_User')
plt.xlim(0,120)
plt.ylim(0,120)

"""TASK: Create a new column based off the difference between critics ratings and users ratings for Rotten Tomatoes. Calculate this with RottenTomatoes-RottenTomatoes_User"""

allsites['RTDIFF'] =allsites['RottenTomatoes']- allsites['RottenTomatoes_User']

allsites['RTDIFF']

"""TASK: Calculate the Mean Absolute Difference between RT scores and RT User scores as described above."""

allsites['RTDIFF'].apply(abs).mean()

"""TASK: Plot the distribution of the differences between RT Critics Score and RT User Score. There should be negative values in this distribution plot. Feel free to use KDE or Histograms to display this distribution."""

sns.histplot(data=allsites,x='RottenTomatoes',y='RottenTomatoes_User')

plt.figure(figsize=(8,5),dpi=100)
sns.histplot(data=allsites, x='RTDIFF',kde=True,bins=30)
plt.title("Distribution of the differences between RT Critics Score and RT User Score")

"""TASK: Now create a distribution showing the absolute value difference between Critics and Users on Rotten Tomatoes."""

plt.figure(figsize=(8,6),dpi=100)
sns.histplot(data=allsites['RTDIFF'].apply(abs),kde=True,bins=20)
plt.title("Abs Difference between RT Critics Score and RT User Score");

"""TASK: What are the top 5 movies users rated higher than critics on average:"""

allsites.sort_values('RTDIFF',ascending=True)[:5][['RottenTomatoes','RottenTomatoes_User','RTDIFF']]

"""TASK: Now show the top 5 movies critics scores higher than users on average."""

allsites.sort_values('RTDIFF',ascending=False)[:5][['RottenTomatoes','RottenTomatoes_User','RTDIFF']]

"""TASK: Display a scatterplot of the Metacritic Rating versus the Metacritic User rating."""

plt.figure(figsize=(6,6),dpi=100)
sns.scatterplot(data=allsites,x='Metacritic',y='Metacritic_User')

"""TASK: Create a scatterplot for the relationship between vote counts on MetaCritic versus vote counts on IMDB."""

plt.figure(figsize=(10,5),dpi=200)
sns.scatterplot(data=allsites,x='Metacritic_user_vote_count', y='IMDB_user_vote_count')

"""TASK: What movie has the highest IMDB user vote count?"""

allsites.sort_values('IMDB_user_vote_count',ascending=False)[:1]

"""TASK: What movie has the highest Metacritic User Vote count?"""

allsites.sort_values('Metacritic_user_vote_count',ascending=False)[:1]

"""TASK: Combine the Fandango Table with the All Sites table. Not every movie in the Fandango table is in the All Sites table, since some Fandango movies have very little or no reviews. We only want to compare movies that are in both DataFrames, so do an inner merge to merge together both DataFrames based on the FILM columns.

"""

df=pd.merge(fandango,allsites, how='inner',on='FILM')

df

df.info()

df.head()

df.describe()

"""TASK: Create new normalized columns for all ratings so they match up within the 0-5 star range shown on Fandango. There are many ways to do this."""

df['RTNorm']=np.round(df['RottenTomatoes']/20,1)
df['RTUNorm']=np.round(df['RottenTomatoes_User']/20,1)

df['MetaNorm']=np.round(df['Metacritic']/20,1)
df['MetaUNorm']=np.round(df['Metacritic_User']/2,1)

df['IMDBNorm']=np.round(df['IMDB']/2,1)

df

df.head()

"""TASK: Now create a norm_scores DataFrame that only contains the normalizes ratings. Include both STARS and RATING from the original Fandango table."""

norm=df[['STARS','RATING','VOTES','RTNorm','RTUNorm','MetaNorm','MetaUNorm','IMDBNorm']]

norm

norm.head()

"""TASK: Create a plot comparing the distributions of normalized ratings across all sites. There are many ways to do this, but explore the Seaborn KDEplot docs for some simple ways to quickly show this. Don't worry if your plot format does not look exactly the same as ours, as long as the differences in distribution are clear."""

def move_legend(ax, new_loc, **kws):
    oldlegend=ax.get_legend()
    handles= oldlegend.legend_handles
    labels= [t.get_text() for t in oldlegend.get_texts()]
    title=oldlegend.get_title().get_text()
    ax.legend(handles,labels, loc=new_loc, title=title, **kws)

fig,ax= plt.subplots(figsize=(15,6),dpi=150)
sns.kdeplot(data=norm, clip=[0,5],fill=True, palette='Set1',ax=ax)
move_legend(ax, "upper left")

"""
TASK: Create a KDE plot that compare the distribution of RT critic ratings against the STARS displayed by Fandango.
"""

fig, ax =plt.subplots(figsize=(15,6),dpi=200)
sns.kdeplot(data=norm[['RTNorm','STARS']], clip=[0,5], fill=True, palette='Set1', ax=ax)
move_legend(ax, "upper left")

norm=norm.drop('VOTES',axis=1)

norm

"""OPTIONAL TASK: Create a histplot comparing all normalized scores."""

plt.subplots(figsize=(15,6),dpi=150)
sns.histplot(norm,bins=50)

"""TASK: Create a clustermap visualization of all normalized scores. Note the differences in ratings, highly rated movies should be clustered together versus poorly rated movies. Note: This clustermap does not need to have the FILM titles as the index, feel free to drop it for the clustermap."""

sns.clustermap(data=norm,cmap='magma',col_cluster=False)

"""TASK: Clearly Fandango is rating movies much higher than other sites, especially considering that it is then displaying a rounded up version of the rating. Let's examine the top 10 worst movies. Based off the Rotten Tomatoes Critic Ratings, what are the top 10 lowest rated movies? What are the normalized scores across all platforms for these movies? You may need to add the FILM column back in to your DataFrame of normalized scores to see the results.


"""

normfile=df[['STARS','RATING','RTNorm','RTUNorm','MetaNorm','MetaUNorm','IMDBNorm','FILM']]

normfile.nsmallest(10,'RTNorm')

"""FINAL TASK: Visualize the distribution of ratings across all sites for the top 10 worst movies."""

plt.figure(figsize=(15,7),dpi=150)
worstfilm=normfile.nsmallest(10,'RTNorm').drop('FILM',axis=1)
sns.kdeplot(data=worstfilm, clip=[0,5], fill=True)

"""THANK YOU"""