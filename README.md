# face_angle_prediction_using_ML-CV
Based on landmark detector and euclidean distance the ML model is trained and which later used to predict the face angle as right, left, up and down.  

<b>Step-1<b><br/>
Download and place <b>shape_predictor_68_face_landmarks.dat<b> in folder from <a href = 'https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2'>here</a> <br/>
<b>Step-2<b><br/>
Unzip data folder and replace data in data folder
<b>Step-3<b><br/>
<pre style="background-color: #eeeeee; border: 1px dashed rgb(153, 153, 153); line-height: 14px; overflow: auto; padding: 5px; width: 100%;"><span style="font-family: &quot;courier new&quot; , &quot;courier&quot; , monospace;"><span style="font-size: 12px;">

#requirments
pip install dlib
pip install opencv-python

</span></span></pre>
<br/>
<h4> To run the predictor model execute following..</h4>
<pre style="background-color: #eeeeee; border: 1px dashed rgb(153, 153, 153); line-height: 14px; overflow: auto; padding: 5px; width: 100%;"><span style="font-family: &quot;courier new&quot; , &quot;courier&quot; , monospace;"><span style="font-size: 12px;">

#open terminal
python main_predict.py
....example <img src="predicted_output.gif" alt="Sample output" />
</span></span></pre>

<br/>
<h4> To prepare new data execute following..</h4>
<pre style="background-color: #eeeeee; border: 1px dashed rgb(153, 153, 153); line-height: 14px; overflow: auto; padding: 5px; width: 100%;"><span style="font-family: &quot;courier new&quot; , &quot;courier&quot; , monospace;"><span style="font-size: 12px;">

<b>open terminal clean data folder</b>
<b>open <i>Data_generation.ipynb</i></b>
</span></span></pre>


<br/>
<h4> To generate the ML model execute following..</h4>
<pre style="background-color: #eeeeee; border: 1px dashed rgb(153, 153, 153); line-height: 14px; overflow: auto; padding: 5px; width: 100%;"><span style="font-family: &quot;courier new&quot; , &quot;courier&quot; , monospace;"><span style="font-size: 12px;">
<b>open <i>preproseing_and_training.ipynb</i></b>
</span></span></pre>

