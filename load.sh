#loading cornell-movie-dataset data
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip -d notebooks
rm cornell_movie_dialogs_corpus.zip
#loading w2v embeddings
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
mkdir -p "embeddings"
gzip "GoogleNews-vectors-negative300.bin.gz" -d embeddings/
rm GoogleNews-vectors-negative300.bin.gz
#loading glove embeddings
wget -c "http://nlp.stanford.edu/data/glove.6B.zip"
unzip "glove.6B.zip" -d embeddings/
rm glove.6B.zip