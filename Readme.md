# Item and SVD based movie recommender system using Django frame
Language: python, html, css, javascript

## Installation

>pip install -r requirements.txt
>

## Run server on terminal
>python manage.py runserver
>

## Open the local website
http://127.0.0.1:8000/

## Algorithm
**Item based** collaborative filtering in this project

**SVD based** SVD in this project using surprise library

The system will choose either one based on the lower RMSE predict score

Recommend movies based on predicting rating, see guidline in [Algorithm- Item_based & SVD.ipynb](https://github.com/xiaolancara/Recommender-System/blob/main/movie_recommender-system/Algorithm-%20Item_based%20%26%20SVD.ipynb)
![image](https://user-images.githubusercontent.com/63172262/115623356-ef22a880-a2ad-11eb-970b-7bcde6f8f7ec.png)

## Function
- user sign up
- user log in
- user rating
- user list library
- movie info and trailer video
- you might like
- search movie
- upload csv file (update movie list)

## Database 
Movie Item: Export csv file from [Imdb](https://www.imdb.com/list/ls022753498/) top 30 popular movies list and added poster and trailer link

Server: db.sqlite3

## Challenge
1. User cold start problem. Sol: I use most rating movies(most popular) to recommend for new user in this project.
2. Movie cold start problem. Movie items that haven't been rated by users will not recommend to any user. Sol: Adding content filtering.
3. The total number of rating is not enough. Since it's item based recommender, it's supposed to have much more number of users than number of items. Thus some users might have 0 similar score items if number of rating in this system is not enough. Sol: In this project, number of ratings more than 100 is ideal.
