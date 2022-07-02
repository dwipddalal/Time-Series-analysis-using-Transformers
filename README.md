# Time Series analysis using Transformers
<a href = 'https://arxiv.org/abs/1706.03762'>‘Attention Is All You Need’ </a>  describes transformers and sequence-to-sequence architecture. It comprises of encoders and decoders that uses Multi-head attention mechannism to generate a fine representation of the input features. 

## Background
<a href='https://www.kaggle.com/competitions/hackrush22-ml-challenge/leaderboard'> Hackrush </a> is an intra-college (IIT Gandhinagar) hackathon in which students of all levels of graduation (B.Tech, M.Tech, Ph.D.) can participate. This year it was conducted on 26th and 27th March. It's a 36hour long hackathon. We(me and my team) <a href='https://www.kaggle.com/competitions/hackrush22-ml-challenge/leaderboard'>won</a> Hackrush this year. 
This is one of the many models that we made for the Hackrush'22 competetion. We made multiple models and performed intergrated stacking on top of these bases models and that's how we were able to secure first rank in this hackathon. To know more about the other models that we used in this contest please refer here.

## Dataset
This dataset was made by Varun Jain and Prof Mayank of IITGN for Hackrush'22.

### Results
Though we used transformers in this case but results were not so pleasing. This because the dataset was very irregular with no general trends sometimes the gap between two adjacent values of dataset was less than 10 and other times it was greater than 500. Also we came to know after the hackathon that the designer of dataset had introduced a lot of noise in the data. So we might get better results if we had removed noise form the dataset as a pre-processing task. Never the less, in this case we observed that though transformer was not good at predicting correct values but it was able to capture overall trend in the dataset. It can be observed from the following image:![image](https://user-images.githubusercontent.com/91228207/161148845-2d29db38-6daf-41f7-8103-8c2c6876652a.png)

