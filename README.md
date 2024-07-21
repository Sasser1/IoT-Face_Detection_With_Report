# A simple python app for logging facial expressions using Deepface
Admittedly, the method is rather outdated as it uses haar cascade, but it was suitable enough for an IoT project we had to do on uni.
The app records your facial expressions and stores them in an Excel spreadsheet, where one spreadsheet shows the detailed log of all expressions recorded, and the other displays a useful summary of percentages of present expressions. This could be useful in fields of psychology or whatever else involves having to keep track of one's facial expressions.

One of the issues could be the fact that the variables are never cleared during the app's running which may eventually slow down the app, and the fact that the spreadsheet is generated only when the app is closed, it may be useful that one has a real time look into the recorded expression so far, perhaps a stat display somewhere in the corner of the app.

