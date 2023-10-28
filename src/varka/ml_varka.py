import pandas as pd
from collections import deque


class MovieLenseVarka:
    """
    create common dataset + split data
    """

    def __init__(self, movies_path, users_path, ratings_path, time_split, train_output_path, val_output_path, max_history_length):
        self._time_split = time_split
        self._max_history_length = max_history_length
        self._train_output_path = train_output_path
        self._val_output_path = val_output_path

        users_names = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        self._users = pd.read_table(users_path, sep='::', header=None, names=users_names, encoding='latin-1', engine='python')
        
        ratings_names = ['UserID', 'MovieID', 'Rating', 'Timestamp']
        self._ratings = pd.read_table(ratings_path, sep='::', header=None, names=ratings_names, encoding='latin-1', engine='python')

        movies_names = ['MovieID', 'Title', 'Genres']
        self._movies = pd.read_table('data/ml-1m/movies.dat', sep='::', header=None, names=movies_names, encoding='latin-1', engine='python')
    
    @staticmethod
    def slice_window(events, max_history_length):
        queue = deque()
        last_timestamp = -1
        ans = []
        events = events.sort_values('Timestamp')
        for movie, ts, title, genre, gender, age, occ, zc, yr, r in zip(events['MovieID'], events['Timestamp'], events['Title'], events['Genres'], events['Gender'], events['Age'], events['Occupation'], events['Zip-code'], events['FilmYear'], events['Rating']):
            queue.append([movie, title, genre, gender, age, occ, zc, yr, ts, r])
            last_timestamp = ts
            if len(queue) > max_history_length + 1:
                queue.popleft()
            if len(queue) > 1: 
                ans.append((list(queue)[:-1], queue[-1], last_timestamp, movie, r, genre))
        return ans

    def do_varka(self):
        data = self._ratings.set_index('MovieID').join(self._movies.set_index('MovieID'), how='left', on='MovieID').reset_index()
        data = data.set_index('UserID').join(self._users.set_index('UserID'), on='UserID', how='left').reset_index()
        data['Genres'] = data['Genres'].apply(lambda x: x.split('|'))
        new_titles = data['Title'].apply(lambda x: x[:-7])
        years = data['Title'].apply(lambda x: int(x[-5:-1]))
        data['Title'] = new_titles
        data['FilmYear'] = years
        data = data.sort_values(by=['Timestamp', 'UserID'])
        data = data.groupby('UserID').apply(lambda x: MovieLenseVarka.slice_window(x, self._max_history_length))
        data = data.explode(0).reset_index()
        data['history'] = data[0].apply(lambda x: x[0])
        data['candidate'] = data[0].apply(lambda x: x[1])
        data['timestamp'] = data[0].apply(lambda x: x[2])
        data['MovieID'] = data[0].apply(lambda x: x[3])
        data['Rating'] = data[0].apply(lambda x: x[4])
        data['Genre'] = data[0].apply(lambda x: x[5])
        data = data[['UserID', 'MovieID', 'Rating', 'history', 'candidate', 'timestamp', 'Genre']].sort_values('timestamp').reset_index()
        train = data[data.timestamp <= self._time_split].sample(frac=1)
        train = train.reset_index()
        train = train[['UserID', 'MovieID', 'Rating', 'history', 'candidate', 'timestamp', 'Genre']]
        val = data[data.timestamp > self._time_split].sample(frac=1)
        val = val.reset_index()
        val = val[['UserID', 'MovieID', 'Rating', 'history', 'candidate', 'timestamp', 'Genre']]

        train.to_pickle(self._train_output_path)
        val.to_pickle(self._val_output_path)
        print('number of rows in train:', train.shape[0], sep=' ')
        print('number of rows in val:', val.shape[0], sep=' ')
