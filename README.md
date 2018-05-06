tdklatt
=======
tdklatt is an open-source implementation of the [Klatt synthesizer][1] in Python 3.6. It was developed for use in the [TrackDraw visual formant synthesizer][2].

People
======
tdklatt was written by Daniel Guest. You can contact him at guest121@umn.edu with questions. The TrackDraw project was written by A.Y. Cho and Daniel Guest.

Usage
=====
Right now, all of the classes and functions of tdklatt are available as a single module (`tdklatt.py`). In the future, it will likely be distributed in a package format instead.

To use tdklatt, all you need to do is clone the repository, and to ensure that you have the dependencies installed. They're listed in the `requirements.txt` file in this repository. To clone the repository, type the following in a terminal:

```
git clone https://github.com/guestdaniel/tdklatt 
```

Alternatively (if you don't use git) you can download the repository in RAR format using the button in the upper right of the repoistory page and unzip it somewhere.

Once you have the repository on your computer, you can hear a quick example synthesized by tdklatt if you just type the following in a terminal to run tdklatt as a script: 

```
python tdklatt.py
```

Here, take care to make sure python points to Python 3.6 on your computer! Earlier versions of Python won't work.

If you want to explore more of tdklatt, launch a Python 3.6 session in the folder containing the repository and run the following:

```python
import tdklatt
```

Now, every class and function provided by tdklatt is available to you in that namespace. To synthesize a speech waveform, try the following:

```python
s = tdklatt.klatt_make() # Creates a Klatt synthesizer w/ default settings
s.run()
```

To play the waveform you just synthesized, try the following:

```python
s.play()
```

Synthesis parameters are stored in `s.params`, which is a standard dictionary. The names of parameters are typically the same as in [the Klatt paper][1], but there may be some exceptions. To see a list of parameters along with explanations, simply access the docstrings of NTVParams1980 (for non-time-varying parameters) and TVParams1980 (for time-varying parameters).

```python
help(tdklatt.NTVParams1980)
help(tdklatt.TVParams1980)
```

To see how different parameters are internally represented, you can call the params dictionary:

```python
s.params
```

To change a parameter, simply modify or replace existing values appropriately. For example, F0 is represented as a numpy.array with length N\_SAMP, so we can replace it with a new numpy array to change the F0.

```python
s.params["F0"] = np.ones(s.params["N_SAMP"])*200 # Change F0 to steady-state 200 Hz
s.run() # Rerun synthesizer
s.play() # Hear the result of the higher F0
```

Currently, only time-varying parameters can be changed in place. Future versions will implement the ability to change non-time-varying parameters as well without re-creating the Synth object.

[1]: http://asa.scitation.org/doi/abs/10.1121/1.383940
[2]: https://github.com/guestdaniel/trackdraw


