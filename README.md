# titanic_prediction
Python script that uses different ML algorithms and Titanic passengers dataset (https://github.com/datasciencedojo/datasets/blob/master/titanic.csv) to determine whether you would survive the Titanic, according to every different algorithm used.

Link to image on dockerhub: https://hub.docker.com/r/davidkri/titanic_prediction/

The container with the script can be pulled using: docker pull davidkri/titanic_prediction

The script can be run using the container with no parameters to see the usage.
The arguments needed by the script are:

- Class: numeric value from 1 to 3
- Sex: 'm' or 'f' for male or female
- Age: numeric value
- Number of Siblings/Spouses Aboard: numeric value
- Number of Parents/Children Aboard: numeric value
- Fare: price of the ticket, numeric value
- Port of Embarkation: 0 for Cherbourg, 1 for Queenstown and 2 for Southampton
