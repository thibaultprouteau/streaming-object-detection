[embed]https://github.com/thibaultprouteau/streaming-object-detection/blob/cd5e9d36029836131171d2c8268ff56853ebb8fb/Poster_D_monstration_F_te_de_la_Science-5.pdf[/embed]


# Install

## Conda Environement

```conda env create -f environment.yml```



## Command to launch streamlit

```streamlit run detection_objet.py --server.address 0.0.0.0 --server.runOnSave true```

## On local machine

```ssh -NL 8501:localhost:8501 host```
