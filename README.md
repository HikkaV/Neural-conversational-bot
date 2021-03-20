# Neural-conversational-bot
For now only notebooks with solution for cornell-movie-dialogues data are available.

<html>
 </head>&nbsp;</head>
<table width="300px">
<tr>
<th colspan="3"><b>In plans</b></th>
</tr>
<tr><th width="350px">Notebooks with raw solution</th>
<td width="50px"><font color="green" size="30">OK</font></td>
</tr>
<tr>
<th>Structured code</th>
<td width="10px"><font color="yellow" size="30">OK</font></td>
</tr>
<tr>
<th>MLFLOW logs tracking</th>
<td width="10px"><font color="yellow" size="30">OK</font></td>
</tr>
<tr>
<th>Various decoding strategies</th>
<td width="10px"><font color="yellow" size="30">OK</font></td>
</tr>
<tr>
<th>Bigger dataset training</th>
<td width="10px"><font color="red" size="30">OK</font></td>
</tr>
<tr>
<th>Bash scripts to load data</th>
<td width="10px"><font color="red" size="30">OK</font></td>
</tr>
<tr>
<th>Telegram bot</th>
<td width="10px"><font color="red" size="30">OK</font></td>
</tr>
</table>
</html>

A neural network was implemented using tensorflow library.

To train a model from scratch:
* Load needed data using `load.sh` script :
`bash load.sh`
* Setup the environment using `setup.sh` script : 
`bash setup.sh`
* Start jupyter notebook (`jupyter notebook`) 
and go on to folder `notebooks`.
* Scripts which start with `processing` prefix - are related 
to data processing and preparation and should be executed before
ones with `training` prefix.
* Training is logged to tensorboard and mlflow.
To see logs of mlflow, make sure that the environment is activated
and run `mlflow ui` from command line.

Trained models and mappings are accessible by the following link:
`https://drive.google.com/file/d/17TXLBbvktU8SP1-9mmtrc9TwENL1i4fX/view?usp=sharing`.

To start telegram bot, run the following command:
` python start.py --path_decoder=<path to decoder> --path_encoder=<path to encoder> --path_mapping=<path to mapping> 
` 
Example of model with filled arguments :
` python start.py --path_decoder=trained_ncb/decoder_all_data_cornell.h5 --path_encoder=trained_ncb/encoder_all_data_cornell.h5 --path_mapping=trained_ncb/token_mapping_cornell.json 
`

This repository is easy to extend by changing the architecture of seq2seq model in `seq2seq.py`.

The work is based on the following papers:

[1] Neural Machine Translation by Jointly Learning to Align and Translate

[2] A Neural Conversational Model

[3] Chameleons in imagined conversations:
A new approach to understanding coordination of linguistic style in dialogs

[4] OpenSubtitles2016: Extracting Large Parallel Corpora
from Movie and TV Subtitles

[5] Glove: Global Vectors for Word Representation

[6] Efficient Estimation of Word Representations in Vector Space

 
