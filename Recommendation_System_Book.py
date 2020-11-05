import pandas as pd



book_data = pd.read_csv("C:\\Users\\nidhchoudhary\\Desktop\\Assignment\\Recommendation_System\\books.csv",encoding = ' ISO-8859-2')

book_data = book_data.rename(columns={"Book.Title":"Title","Book.Author":"Author"})


book_data["Author"] = book_data["Author"].fillna(" ")

from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(stop_words="english")  
tfdif_data= book_data['Author']
tfdif_metric = tfidf.fit_transform(book_data.Author)


from sklearn.metrics.pairwise import linear_kernel
cosine_metric = linear_kernel(tfdif_metric,tfdif_metric)

#print(cosine_metric)

#anime_index = pd.Series(anime.index,index=anime['name']).drop_duplicates()



df_index=pd.Series(book_data.index,index=book_data['Title']).drop_duplicates()

def get_title_recommendations(Title,topN):

    #topN = 10
    # Getting the movie index using its title 
    df_id = df_index[Title]
    
    # Getting the pair wise similarity score for all the df's with that 
    # df
    cosine_scores = list(enumerate(cosine_metric[df_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar df's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the df index 
    df_idx  =  [i[0] for i in cosine_scores_10]
    df_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    df_similar_show = pd.DataFrame(columns=["Title","Score"])
    df_similar_show["Title"] = book_data.loc[df_idx,"Title"]
    df_similar_show["Score"] = df_scores
    df_similar_show.reset_index(inplace=True)  
    df_similar_show.drop(["index"],axis=1,inplace=True)
    print (df_similar_show)
    #return (df_similar_show)


get_title_recommendations('Nights Below Station Street',topN = 15)