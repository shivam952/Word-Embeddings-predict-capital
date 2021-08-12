import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import get_vectors

data = pd.read_csv('capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']


word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
len(word_embeddings) 

## Predict someething among the words

#The function will take as input three words.
#The first two are related to each other.
#It will predict a 4th word which is related to the third word in a similar manner as the two first words are related to each other.
#As an example, "Athens is to Greece as Bangkok is to __"?
#You will write a program that is capable of finding the fourth word.

# the concept of cosine similarity is used

#function to calculate cosine similarity : 
def cosine_similarity(A,B):
     
    dot = np.dot(A,B)
    norma = np.sqrt(np.dot(A,A))
    normb = np.sqrt(np.dot(B,B))
    cos = dot / (norma*normb)
    
    return cos

king = word_embeddings['king']
queen = word_embeddings['queen']

print(cosine_similarity(king, queen))

#A cosine value between 0 and 1 shows similarity

#Function that computes euclidean distance between two vectors
def euclidean(A,B):
    d = np.linalg.norm(A-B)
    return d

# Now we will try to find the country for given capital
#input : Athens, Greece, Baghdad #output: Iraq

def get_country(city1, country1,city2,embeddings):
    # store the city1, country 1, and city 2 in a set called group
    group = set((city1, country1, city2))
    
    # get embeddings of city 1
    city1_emb = word_embeddings[city1]

    # get embedding of country 1
    country1_emb =  word_embeddings[country1]

    # get embedding of city 2
    city2_emb = word_embeddings[city2]
    
    vec = country1_emb - city1_emb + city2_emb
    
    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1

    # initialize country to an empty string
    country = ''
    
    for word in word_embeddings.keys():
        if word not in group:
            word_emb = word_embeddings[word]
            
            cur_similarity = cosine_similarity(vec,word_emb)
            
            if cur_similarity > similarity:

                # update the similarity to the new, better similarity
                similarity = cur_similarity

                # store the country as a tuple, which contains the word and the similarity
                country = (word, similarity)
    return country

# Testing the function, note to make it more robust you can return the 5 most similar words.
get_country('Athens', 'Greece', 'Cairo', word_embeddings)
# output is egypt 0.76

#lets test the accuracy of the model
def get_accuracy(embeddings, data):
    num_correct = 0
    
    for i,row in data.iterrows():
        city1 = row['city1']
        country1 = row['country1']
        city2 = row['city2']
        country2 = row['country2']
        
        predicted_country2, _ = get_country(city1,country1,city2,word_embeddings)
        
        if predicted_country2 == country2:
            num_correct += 1
    
    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)

    # calculate the accuracy by dividing the number correct by m
    accuracy = num_correct/m

    
    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")        


#Principal component analysis

def compute_pca(X,n_components=2):
    # mean center the data
    X_demeaned = X - np.mean(X,axis=0)
    print('X_demeaned.shape: ',X_demeaned.shape)
    
    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix, UPLO='L')
    #print('Eigen vectors shape: ',eigen_vals.shape)
    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)
    
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = idx_sorted[::-1]
    
    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]
    
    eigen_vecs_subset = eigen_vecs_sorted[:,0:n_components]
    
    print('Eigen vectors shape: ',eigen_vecs_subset.shape)
    
    X_reduced = np.dot(eigen_vecs_subset.transpose(),X_demeaned.transpose()).transpose()
    
    return X_reduced


# Testing your function
np.random.seed(1)
X = np.random.rand(3, 10)
#print(X)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# given a list of words and the embeddings, it returns a matrix with all the embeddings
X = get_vectors(word_embeddings, words)

print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)

result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()


     
    
           

