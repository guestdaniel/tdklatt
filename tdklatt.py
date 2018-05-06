"""
tdklatt.py

Provides a number of functions and classes which enable the user to synthesize
speech waveforms ala Klatt 1980. Currently supports the Klatt 1980 algorithm at
10 kHz sampling rate.

Classes:
    KlattParam1980: Object containing all paramteres for Klatt synthesizer
    KlattSynth: Top-level KlattSynth object.
    KlattSection: Section-level synthesis object (e.g., voicing source or
        filter cascade)
    KlattComponent: Component-level synthesis object (e.g., filter or
        amplifier)

Functions:
    klatt_make: Creates a KlattSynth object and properly initializes it using
        a KlattParam1980 object

Examples:
    Create a /pa/ and plot a figure showing the spectrogram and time waveform.
    >>> python tdklatt.py

    Create a synthesizer with default settings, run it, and hear the output.
    >>> python
    >>> from tdklatt import *
    >>> s = klatt_make()
    >>> s.run()
    >>> s.play()

    Create a synthesizer, change the F0, and hear the output.
    >>> python
    >>> from tdklatt import *
    >>> s = klatt_make()
    >>> s.params["F0"] = np.ones(s.params["N_SAMP"])*200 # Change F0 to 200 Hz
    >>> s.run()
    >>> s.play()
"""

try:
    import math
    import numpy as np
    from scipy.signal import resample_poly
    from scipy.io.wavfile import write
    import simpleaudio as sa
except ImportError:
    print("Missing one or more required modules.")
    print("Please make sure that math, numpy, scipy, and simpleaudio are installed.")
    import sys
    sys.exit()

def klatt_make(params=None):
    """
    Creates and prepares a KlattSynth object.

    The user can provide a KlattParam1980 object to the params argument
    if they wish to initialize those objects and modify parameter values
    first. If none is provided, then a default param object is initialized
    and used. First, the KlattSynth object is
    initialized, and then each parameter passed to the KlattSynth object.

    Arguments:
        params (KlattParam1980): parameters object

    Returns:
        synth (KlattSynth): Klatt synthesizer object, ready to run()
    """
    # Choose defaults if custom parameters not available
    if params is None:
        params = KlattParam1980()
    # Initialize synth
    synth = KlattSynth()

    # Loop through all time-varying parameters, processing as needed
    for param in list(filter(lambda aname: not aname.startswith("_"),
                             dir(params))):
        if param is "FF" or param is "BW":
            synth.params[param] = \
                    [getattr(params, param)[i] for i in range(params.N_FORM)]
        else:
            synth.params[param] = getattr(params, param)
    synth.setup()
    return(synth)


class KlattParam1980(object):
    """
    Class container for parameters for Klatt 1980 synthesizer.

    Arguments:
        F0 (float): Fundamental frequency in Hz
        FF (list): List of floats, each one corresponds to a formant frequency
            in Hz
        BW (list): List of floats, each one corresponds to the bandwidth of a
            formant in Hz in terms of plus minus 3dB
        AV (float): Amplitude of voicing in dB
        AVS (float): Amplitude of quasi-sinusoidal voicing in dB
        AH (float): Amplitude of aspiration in dB
        AF (float): Amplitude of frication in dB
        SW (0 or 1): Controls switch from voicing waveform generator to cascade
            or parallel resonators
        FGP (float): Frequency of the glottal resonator 1 in Hz
        BGP (float): Bandwidth of glottal resonator 1 in Hz
        FGZ (float): Frequency of glottal zero in Hz
        BGZ (float): Bandwidth of glottal zero in Hz
        FNP (float): Frequency of nasal pole in Hz
        BNP (float): Bandwidth of nasal pole in Hz
        FNZ (float): Frequency on the nasal zero in Hz
        BNZ (float): Bandwidth of nasal zero in Hz
        BGS (float): Glottal resonator 2 bandwidth in Hz
        A1 (float): Amplitude of parallel formant 1 in Hz
        A2 (float): Amplitude of parallel formant 2 in Hz
        A3 (float): Amplitude of parallel formant 3 in Hz
        A4 (float): Amplitude of parallel formant 4 in Hz
        A5 (float): Amplitude of parallel formant 5 in Hz
        A6 (float): Amplitude of parallel formant 6 in Hz
        AN (float): Amplitude of nasal formant in dB

    Attributes:
        Each of the above time-varying parameters is stored as an attribute in
            the form of a Numpy array.
    """
    def __init__(self, FS=10000, N_FORM=5, DUR=1, F0=100,
                       FF=[500, 1500, 2500, 3500, 4500],
                       BW=[50, 100, 100, 200, 250],
                       AV=60, AVS=0, AH=0, AF=0,
                       SW=0, FGP=0, BGP=100, FGZ=1500, BGZ=6000,
                       FNP=250, BNP=100, FNZ=250, BNZ=100, BGS=200,
                       A1=0, A2=0, A3=0, A4=0, A5=0, A6=0, AN=0):
        self.FS = FS
        self.DUR = DUR
        self.N_FORM = N_FORM
        self.N_SAMP = round(FS*DUR)
        self.VER = "KLSYN80"
        self.DT = 1/FS
        self.F0 = np.ones(self.N_SAMP)*F0
        self.FF = [np.ones(self.N_SAMP)*FF[i] for i in range(N_FORM)]
        self.BW = [np.ones(self.N_SAMP)*BW[i] for i in range(N_FORM)]
        self.AV = np.ones(self.N_SAMP)*AV
        self.AVS = np.ones(self.N_SAMP)*AVS
        self.AH = np.ones(self.N_SAMP)*AH
        self.AF = np.ones(self.N_SAMP)*AF
        self.FNZ = np.ones(self.N_SAMP)*FNZ
        self.SW = np.ones(self.N_SAMP)*SW
        self.FGP = np.ones(self.N_SAMP)*FGP
        self.BGP = np.ones(self.N_SAMP)*BGP
        self.FGZ = np.ones(self.N_SAMP)*FGZ
        self.BGZ = np.ones(self.N_SAMP)*BGZ
        self.FNP = np.ones(self.N_SAMP)*FNP
        self.BNP = np.ones(self.N_SAMP)*BNP
        self.BNZ = np.ones(self.N_SAMP)*BNZ
        self.BGS = np.ones(self.N_SAMP)*BGS
        self.A1 = np.ones(self.N_SAMP)*A1
        self.A2 = np.ones(self.N_SAMP)*A2
        self.A3 = np.ones(self.N_SAMP)*A3
        self.A4 = np.ones(self.N_SAMP)*A4
        self.A5 = np.ones(self.N_SAMP)*A5
        self.A6 = np.ones(self.N_SAMP)*A6
        self.AN = np.ones(self.N_SAMP)*AN


class KlattSynth(object):
    """
    Synthesizes speech ala Klatt 1980 and Klatt 1990.

    Attributes:
        name (string): Name of this synthesizer
        output (None): Output vector for the synthesizer, set by setup()
            later
        sections (None): List of sections in the synthesizer, set by
            setup() later
        params (dictionary): Dictionary of parameters, all entries are
            initialized as None and are set later

    Methods:
        setup: Run after parameter values are set, initializes synthesizer
        run: Clears current output vector and runs synthesizer
        play: Plays output via sounddevice module

    KlattSynth contains all necessary synthesis parameters in an attribute
    called params. The synthesis routine is organized around the concept of
    sections and components. Sections are objects which represent organizational
    abstractions drawn from the Klatt 1980 paper. Each section is composed of
    multiple components, which are small signal processing units like
    individual filters, resonators, amplifiers, etc. Each section has a run()
    method with performs the operation that section is designed to do. For
    example, a KlattVoice section's run() method generates a voicing waveform.

    KlattSynth's params attribute will need to be provided with parameters, there
    are no built-in defaults! Currently the only way to do so is to generate a
    KlattSynth object through the klatt_make function available in interface.py
    or to do so manually. params is just a dictionary that can be directly
    addressed, so setting parameters is easy. One important caveat is that any
    time-varying parameters (i.e., all parameters except those labelled
    "synth settings" below) should be numpy arrays of length N_SAMP.

    KlattSynth, while designed around the original Klatt formant synthesizer,
    is designed to be flexible and modular. It supports multiple versions of
    the Klatt synthesis routine and is easily extensible, so you easily add
    custom components and sections to suit your needs.

    Current supported synthesis routines:
        - Klatt 1980, "KLSYN80"
            Klatt, D. H. (1980). Software for a cascade/parallel formant
            synthesizer. The Journal of the Acoustical Society of America,
            67 (3)
    """
    def __init__(self):
        """
        Initializes KlattSynth object.

        Creates name and tag which can be used by TrackDraw to display current
        synthesis type. Also creates the parameter list, but leaves it blank.
        """
        # Create name
        self.name = "Klatt Formant Synthesizer"

        # Create empty attributes
        self.output = None
        self.sections = None

        # Create synthesis parameters dictionary
        param_list = ["F0", "AV", "OQ", "SQ", "TL", "FL", # Source
                      "DI", "AVS", "AV", "AF", "AH",      # Source
                      "FF", "BW",                         # Formants
                      "FGP", "BGP", "FGZ", "BGZ", "BGS",  # Glottal pole/zero
                      "FNP", "BNP", "FNZ", "BNZ",         # Nasal pole/zero
                      "FTP", "BTP", "FTZ", "BTZ",         # Tracheal pole/zero
                      "A2F", "A3F", "A4F", "A5F", "A6F",  # Frication parallel
                      "B2F", "B3F", "B4F", "B5F", "B6F",  # Frication parallel
                      "A1V", "A2V", "A3V", "A4V", "ATV",  # Voicing parallel
                      "A1", "A2", "A3", "A4", "A5", "AN", # 1980 parallel
                      "ANV",                              # Voicing parallel
                      "SW", "INV_SAMP",                   # Synth settings
                      "N_SAMP", "FS", "DT", "VER"]        # Synth settings
        self.params = {param: None for param in param_list}

    def setup(self):
        """
        Sets up KlattSynth.

        Run after parameter values are set. Initializes output vector initializes
        sections and sets them to be attributes.

        NOTE: it's probably bad practice to create new attributes outside of
        __init__(), but this is the approach for now...
        """
        # Initialize data vectors
        self.output = np.zeros(self.params["N_SAMP"])

        # Differential functiontioning based on version...
        if self.params["VER"] == "KLSYN80":
            # Initialize sections
            self.voice = KlattVoice1980(self)
            self.noise = KlattNoise1980(self)
            self.cascade = KlattCascade1980(self)
            self.parallel = KlattParallel1980(self)
            self.radiation = KlattRadiation1980(self)
            self.output_module = OutputModule(self)
            # Create section-level connections
            self.voice.connect([self.cascade, self.parallel])
            self.noise.connect([self.cascade, self.parallel])
            self.cascade.connect([self.radiation])
            self.parallel.connect([self.radiation])
            self.radiation.connect([self.output_module])
            # Put all section objects into self.sections for reference
            self.sections = [self.voice, self.noise, self.cascade,
                             self.parallel, self.radiation, self.output_module]
            # Patch all components together within sections
            for section in self.sections:
                section.patch()
        else:
            print("Sorry, versions other than Klatt 1980 are not supported.")

    def run(self):
        """
        Runs KlattSynth.

        Sets output to zero, then runs each component before extracting output
        from the final section (output_module).
        """
        self.output[:] = np.zeros(self.params["N_SAMP"])
        # Clear inputs and outputs in each component
        for section in self.sections:
            for component in section.components:
                component.clean()
        for section in self.sections:
            section.run()
        self.output[:] = self.output_module.output[:]

    def _get_int16at16K(self):
        """
        Transforms output waveform to form amenable for playing/saving.
        """
        assert self.params["FS"] == 10_000
        y = resample_poly(self.output, 8, 5)  # resample from 10K to 16K
        maxabs = np.max(np.abs(y))
        if maxabs > 1:
            y /= maxabs
        y = np.round(y * 32767).astype(np.int16)
        return y

    def play(self):
        """
        Plays output waveform.
        """
        y = self._get_int16at16K()
        sa.play_buffer(y, num_channels=1, bytes_per_sample=2, sample_rate=16_000)

    def save(self, path):
        """
        Saves output waveform to disk.

        Arguments:
            path (str): where the file should be saved
        """
        y = self._get_int16at16K()
        write(path, 16_000, y)


##### CLASS DEFINITIONS #####
class KlattSection:
    """
    Parent class for section-level objects in TrackDraw synth system.

    Arguments:
        mast (KlattSynth): Master KlattSynth object, allows all
            sub-components to access params directly

    Attributes:
        mast (KlattSynth): see Arguments
        ins (list): list of Buffer objects for handling this Section's
            inputs, if it has any
        outs (list): list of Buffer objects for handling this Section's
            outputs, if it has any

    Methods:
        connect: Used to connect two sections
        process_ins: Processes all input buffers
        process_outs: Processes all output buffers
        run: Calls self.do(), which processes the signal by calling components'
            methods as necessary

    An operational Section needs two custom methods to be implemented on top of
    the default methods provided by the class definition:
        1) patch(), which describes how components should be connected, and
        2) do(), which describes the order in which components should be run
            and what parameters they should use during their operation

    patch() is called by KlattSynth.setup(), while do() is called by this
    class's generic run() method
    """
    def __init__(self, mast):
        self.mast = mast
        self.components = []
        self.ins = []
        self.outs = []

    def connect(self, sections):
        """
        Connects this section to another.

        Arguments:
            sections (list): list of Section objects to be connected to this
                Section

        For each Section in sections, this method appends a Buffer to that
        Section's outs and another Buffer to this section's ins. It also
        connects the two so that signals propogate between them. See the doc
        strings for the Component level operations of connect, send, and
        receive to understand more about how this signal propogation occurs.
        """
        for section in sections:
            section.ins.append(Buffer(mast=self.mast))
            self.outs.append(Buffer(mast=self.mast, dests=[section.ins[-1]]))

    def process_ins(self):
        """
        Processes all input buffers.

        Calls the Buffer's process() method for each Buffer in this section's
        ins.
        """
        for _in in self.ins:
            _in.process()

    def process_outs(self):
        """
        Processes all output buffers.

        Calls the Buffer's process() method for each Buffer in this section's
        outs.
        """
        for out in self.outs:
            out.process()

    def run(self):
        """
        Carries out processing of this Section.

        If ins is not empty, processes ins. Then calls this Section's custom
        do() method. Then, if outs is not empty, processes outs.
        """
        if self.ins is not None:
            self.process_ins()
        self.do()
        if self.outs is not None:
            self.process_outs()


class KlattComponent:
    """
    Parent class for component-level objects in TrackDraw synth system.

    Arguments:
        mast (KlattSynth): master KlattSynth object, allows for access to
            params
        dests (list): list of other Components, see send() method doc string
            for more information on how this list is used

    Attributes:
        mast (KlattSynth): see Arguments
        dests (list): see Arguments
        input (Numpy array): input vector, length N_SAMP
        output (Numpy array): output vector, length N_SAMP

    Methods:
        receive: Changes input vector
        send: Propagates output to all destinations
        connect: Used to connect two components together

    Components are small signal processing units (e.g., filters, amplifiers)
    which compose Sections. Components are connected at the Section level using
    the Components' connect() method. Components use the send() and receive()
    methods to perpetuate the signal down the signal chain.

    Components should all have a custom method implemented in which, at the
    very least, some output is put in the output attribute and, at the end of
    processing, the send() method is called.
    """
    def __init__(self, mast, dests=None):
        self.mast = mast
        if dests is None:
            self.dests = []
        else:
            self.dests = dests
        self.input = np.zeros(self.mast.params["N_SAMP"])
        self.output = np.zeros(self.mast.params["N_SAMP"])

    def receive(self, signal):
        """
        Updates current signal.

        Arguments:
            signal (NumPy array): vector to change input to

        Used by the send() method to propogate the signal through the chain of
        components, changes this Component's input attribute to be equal to the
        signal argument.
        """
        self.input[:] = signal[:]

    def send(self):
        """
        Perpetuates signal to components further down in the chain.

        For each Component in this Component's dests list, uses the
        receive() method to set that Component's input to this Component's
        output, thereby propagating the signal through the chain of components.

        NOTE: Mixer has a custom implementation of receive, but it interfaces
        identically, so you don't need to worry about it.
        """
        for dest in self.dests:
            dest.receive(signal=self.output[:])

    def connect(self, components):
        """
        Connects two components together.

        Arguments:
            components (list): list of Components to be connected to

        For each destination Component in the list components, adds the
        destination Component to this Component's dests.
        """
        for component in components:
            self.dests.append(component)

    def clean(self):
        self.input = np.zeros(self.mast.params["N_SAMP"])
        self.output = np.zeros(self.mast.params["N_SAMP"])

##### SECTION DEFINITIONS #####
class KlattVoice1980(KlattSection):
    """
    Generates a voicing waveform ala Klatt 1980.

    Passes an impulse train with time-varying F0 through a series of filters
    to generate both normal and quasi-sinusoidal voicing waveforms. Then,
    amplifies and mixes the two waveforms. Passes the mixed output onward
    through a time-varying binary switch.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        impulse (Impulse): Periodic pulse train generator with fundamental
            frequency F0
        rgp (Resonator): Glottal pole resonator to generate normal voicing
            waveform with center frequency FGP and bandwidth BGP
        rgz (Resonator): Glottal zero antiresonator with center frequency FGZ
            bandwidth BGZ
        rgs (Resonator): Secondary glottal resonator to generate
            quasi-sinusoidal voicing waveform with center frequency FGP and
            bandwidth BGS
        av (Amplifier): Amplifier to control amplitude of normal voicing with
            amplification amount AV
        avs (Amplifier): Amplifier to control amplitude of quasi-sinuosidal
            voicing with amplification amount AVS
        mixer (Mixer): Mixer to mix normal voicing waveform and
            quasi-sinusoidal voicing waveforms
        switch (Switch): Switch to switch destination of KlattVoice1980 to
            cascade filter track (SW=0) or parallel filter track with (SW=1)
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.impulse = Impulse(mast=self.mast)
        self.rgp = Resonator(mast=self.mast)
        self.rgz = Resonator(mast=self.mast, anti=True)
        self.rgs = Resonator(mast=self.mast)
        self.av = Amplifier(mast=self.mast)
        self.avs = Amplifier(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.switch = Switch(mast=self.mast)
        self.components = [self.impulse, self.rgp, self.rgz, self.rgs, \
                           self.av, self.avs, self.mixer, self.switch]

    def patch(self):
        self.impulse.connect([self.rgp])
        self.rgp.connect([self.rgz, self.rgs])
        self.rgz.connect([self.av])
        self.rgs.connect([self.avs])
        self.av.connect([self.mixer])
        self.avs.connect([self.mixer])
        self.mixer.connect([self.switch])
        self.switch.connect([*self.outs])

    def do(self):
        self.impulse.impulse_gen(F0=self.mast.params["F0"])
        self.rgp.resonate(ff=self.mast.params["FGP"],
                          bw=self.mast.params["BGP"])
        self.rgz.resonate(ff=self.mast.params["FGZ"],
                          bw=self.mast.params["BGZ"])
        self.rgs.resonate(ff=self.mast.params["FGP"],
                          bw=self.mast.params["BGS"])
        self.av.amplify(dB=self.mast.params["AV"])
        self.avs.amplify(dB=self.mast.params["AVS"])
        self.mixer.mix()
        self.switch.operate(choice=self.mast.params["SW"])


class KlattNoise1980(KlattSection):
    """
    Generates noise ala Klatt 1980.

    Generates Gaussian noise which is then low-pass filtered and amplified.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        noisegen (Noisegen): Gaussian noise generator
        lowpass (Lowpass): Simple time-domain lowpass filter
        amp (Amplifier): Amplifier to control amplitude of lowpassed noise
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.noisegen = Noisegen(mast=self.mast)
        self.lowpass = Lowpass(mast=self.mast)
        self.amp = Amplifier(mast=self.mast)
        self.components = [self.noisegen, self.lowpass, self.amp]

    def patch(self):
        self.noisegen.connect([self.lowpass])
        self.lowpass.connect([self.amp])
        self.amp.connect([*self.outs])

    def do(self):
        self.noisegen.generate()
        self.lowpass.filter()
        self.amp.amplify(dB=-60)  # TODO: Need to figure out a real value


class KlattCascade1980(KlattSection):
    """
    Simulates a vocal tract with a cascade of resonators.

    Passes noise waveform (ins[1]) through amplifier, and then mixes it with
    voicing waveform (ins[0]). The mixed waveform is then passed through
    resonators to simulate the nasal pole and zero and then through a cascade
    of resonators to introduce formants 1-5 into the sound.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        ah (Amplifier): Amplifier to control amplitude of noise waveform with
            amplification amount AH
        mixer (Mixer): Mixer to mix voicing and noise waveforms
        rnp (Resonator): Resonator to create nasal pole with center frequency
            FNP and bandwidth BNP
        rnz (Resonator): Antiresonator to create nasal zero with center
            frequency FNZ and bandwdith BNZ
        formants (list): List of Resonators to introduce formants into the
            spectrum, contains N_FORM formants. Formant frequency values are
            in the param FF, which is a list of arrays where the n-th array
            contains the n-th formant frequency values. Bandwidth values are in
            the param BW, which is a list of arrays where the n-th array
            contains the n-th formant's bandwidth values.
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.ah = Amplifier(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.rnp = Resonator(mast=self.mast)
        self.rnz = Resonator(mast=self.mast, anti=True)
        self.formants = []
        for form in range(self.mast.params["N_FORM"]):
            self.formants.append(Resonator(mast=self.mast))
        self.components = [self.ah, self.mixer, self.rnp, self.rnz] + \
            self.formants

    def patch(self):
        self.ins[0].connect([self.mixer])
        self.ins[1].connect([self.ah])
        self.ah.connect([self.mixer])
        self.mixer.connect([self.rnp])
        self.rnp.connect([self.rnz])
        self.rnz.connect([self.formants[0]])
        for i in range(0, self.mast.params["N_FORM"]-1):
            self.formants[i].connect([self.formants[i+1]])
        self.formants[self.mast.params["N_FORM"]-1].connect([*self.outs])

    def do(self):
        self.ah.amplify(dB=self.mast.params["AH"])
        self.mixer.mix()
        self.rnp.resonate(ff=self.mast.params["FNP"],
                          bw=self.mast.params["BNP"])
        self.rnz.resonate(ff=self.mast.params["FNZ"],
                          bw=self.mast.params["BNZ"])
        for form in range(len(self.formants)):
            self.formants[form].resonate(ff=self.mast.params["FF"][form],
                                         bw=self.mast.params["BW"][form])


class KlattParallel1980(KlattSection):
    """
    Simulates a vocal tract with a bank of parallel resonators.

    Directs the noise waveform to an amplifier, and the voicing waveform to a
    differentiator (highpass filter). Passes the high-passed voicing waveform
    and amplified noise waveform to amplifier-resonator pairs which correspond
    to the nasal formant and to formants 2-4. Passses the un-altered voicing
    waveform to an amplifier-resonator pair which corresponds to formant 1.
    Passes the amplified noise waveform to amplifier-resonator pairs which
    correspond to formants 5-6 and to a bypass path. Mixes the output of all
    the resonators and the bypass path.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        af (Amplifier): Amplifier to control the amplitude of the noise
            waveform (ins[1]) with amplification amount AF
        first_diff (Firstdiff): First differentiator (highpass filter)
        mixer (Mixer): Mixer to mix amplified noise waveform and highpass
            filtered voicing waveform
        an (Amplifier): Amplifier to control the amplitude of the nasal formant
            with amplification amount AN
        rnp (Resonator): Resonator to create the nasal formant, with center
            frequency FNP and bandwidth BNP
        a1 (Amplifier): Amplifier to control the amplitude of the first formant
            with amplification amount A1
        r1 (Resonator): Resonator to create the first formant, with center
            frequency and bandwidth in the 0-th array of FF and BW respectively
        a2 (Amplifier): Amplifier to control the amplitude of the first formant
            with amplification amount A2
        r2 (Resonator): Resonator to create the second formant, with center
            frequency and bandwidth in the 1-th array of FF and BW respectively
        a3 (Amplifier): Amplifier to control the amplitude of the second formant
            with amplification amount A3
        r3 (Resonator): Resonator to create the third formant, with center
            frequency and bandwidth in the 2-th array of FF and BW respectively
        a4 (Amplifier): Amplifier to control the amplitude of the third formant
            with amplification amount A4
        r4 (Resonator): Resonator to create the fourth formant, with center
            frequency and bandwidth in the 3-th array of FF and BW respectively
        a5 (Amplifier): Amplifier to control the amplitude of the fifth formant
            with amplification amount A5
        r5 (Resonator): Resonator to create the fifth formant, with center
            frequency and bandwidth in the 4-th array of FF and BW respectively
        a6 (Amplifier): Amplifier to control the amplitude of the sixth formant
            with amplification amount A6
        r6 (Resonator): Resonator to create the sixth formant, with center
            frequency and bandwidth in the 5-th array of FF and BW respectively
        ab (Amplifier): Amplifier to control the amplitude of the bypass path
            with amplification amount ?? (see comments below)
        output_mixer (Mixer): Mixer to mix various formant waveforms and bypass
            path output
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.af = Amplifier(mast=self.mast)
        self.a1 = Amplifier(mast=self.mast)
        self.r1 = Resonator(mast=self.mast)
        self.first_diff = Firstdiff(mast=self.mast)
        self.mixer = Mixer(mast=self.mast)
        self.an = Amplifier(mast=self.mast)
        self.rnp = Resonator(mast=self.mast)
        self.a2 = Amplifier(mast=self.mast)
        self.r2 = Resonator(mast=self.mast)
        self.a3 = Amplifier(mast=self.mast)
        self.r3 = Resonator(mast=self.mast)
        self.a4 = Amplifier(mast=self.mast)
        self.r4 = Resonator(mast=self.mast)
        self.a5 = Amplifier(mast=self.mast)
        self.r5 = Resonator(mast=self.mast)
        # TODO: 6th formant currently not part of self.do()! Not sure what values
        # to give to it... need to keep reading Klatt 1980.
        self.a6 = Amplifier(mast=self.mast)
        self.r6 = Resonator(mast=self.mast)
        # TODO: ab currently not part of self.do()! Not sure what values to give
        # to it... need to keep reading Klatt 1980.
        self.ab = Amplifier(mast=self.mast)
        self.output_mixer = Mixer(mast=self.mast)
        self.components = [self.af, self.a1, self.r1, self.first_diff, \
                           self.mixer, self.an, self.rnp, self.a2, self.r2, \
                           self.r1, self.first_diff, self.mixer, self.an, \
                           self.rnp, self.a2, self.r2, self.a3, self.r3, \
                           self.a4, self.r4, self.a5, self.r5, self.a6, \
                           self.r6, self.ab, self.output_mixer]

    def patch(self):
        self.ins[1].connect([self.af])
        self.ins[0].connect([self.a1, self.first_diff])
        self.af.connect([self.mixer, self.a5, self.a6, self.ab])
        self.first_diff.connect([self.mixer])
        self.mixer.connect([self.an, self.a2, self.a3, self.a4])
        self.a1.connect([self.r1])
        self.an.connect([self.rnp])
        self.a2.connect([self.r2])
        self.a3.connect([self.r3])
        self.a4.connect([self.r4])
        self.a5.connect([self.r5])
        self.r6.connect([self.r6])
        for item in [self.r1, self.r2, self.r3, self.r4, self.r5, \
                     self.r6, self.rnp, self.ab]:
            item.connect([self.output_mixer])
        self.output_mixer.connect([*self.outs])

    def do(self):
        self.af.amplify(dB=self.mast.params["AF"])
        self.a1.amplify(dB=self.mast.params["A1"])
        self.r1.resonate(ff=self.mast.params["FF"][0],
                         bw=self.mast.params["BW"][0])
        self.first_diff.differentiate()
        self.mixer.mix()
        self.an.amplify(dB=self.mast.params["AN"])
        self.rnp.resonate(ff=self.mast.params["FNP"],
                          bw=self.mast.params["BNP"])
        self.a2.amplify(dB=self.mast.params["A2"])
        self.r2.resonate(ff=self.mast.params["FF"][1],
                         bw=self.mast.params["BW"][1])
        self.a3.amplify(dB=self.mast.params["A3"])
        self.r3.resonate(ff=self.mast.params["FF"][2],
                         bw=self.mast.params["BW"][2])
        self.a4.amplify(dB=self.mast.params["A4"])
        self.r4.resonate(ff=self.mast.params["FF"][3],
                         bw=self.mast.params["BW"][3])
        self.a5.amplify(dB=self.mast.params["A5"])
        self.r5.resonate(ff=self.mast.params["FF"][4],
                         bw=self.mast.params["BW"][4])
        self.output_mixer.mix()


class KlattRadiation1980(KlattSection):
    """
    Simulates the effect of radiation characteristic in the vocal tract.

    Simply mixes inputs, and then highpass filters them (via calculating
    the first derivative).

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        mixer (Mixer): Mixer to mix various inputs
        firstdiff (Firstdiff): First differentiator to act as highpass filter
            (in this case, models the effect of the radiation characteristic of
            the lips)
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.mixer = Mixer(mast=self.mast)
        self.firstdiff = Firstdiff(mast=self.mast)
        self.components = [self.mixer, self.firstdiff]

    def patch(self):
        for _in in self.ins:
            _in.connect([self.mixer])
        self.mixer.connect([self.firstdiff])
        self.firstdiff.connect([*self.outs])

    def do(self):
        self.mixer.mix()
        self.firstdiff.differentiate()


class OutputModule(KlattSection):
    """
    Mixes inputs and then normalizes mixed waveform by setting peak value of 1.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        mixer (Mixer): Mixer to mix various inputs
        normalizer (Normalizer): Divides waveform by its absolute value maximum
        output (np.array): Final destination for synthesized speech waveform,
            extracted by KlattSynth object after synthesis is complete
    """
    def __init__(self, mast):
        KlattSection.__init__(self, mast)
        self.mixer = Mixer(mast=self.mast)
        self.normalizer = Normalizer(mast=self.mast)
        self.output = np.zeros(self.mast.params["N_SAMP"])
        self.components = [self.mixer, self.normalizer]

    def patch(self):
        for _in in self.ins:
            _in.dests = [self.mixer]
        self.mixer.dests = [self.normalizer]
        self.normalizer.dests = [*self.outs]

    def do(self):
        self.mixer.mix()
        self.normalizer.normalize()
        self.output[:] = self.normalizer.output[:]


##### COMPONENT DEFINITIONS #####
class Buffer(KlattComponent):
    """
    Utility component used in signal propagation.

    Arguments:
        mast (KlattSynth): see parent class
        dests (None): see parent class
    """
    def __init__(self, mast, dests=None):
        KlattComponent.__init__(self, mast, dests)

    def process(self):
        """
        Sets output to be the input waveform, and then sends output waveform to
        downstream connected components.
        """
        self.output[:] = self.input[:]
        self.send()


class Resonator(KlattComponent):
    """
    Klatt resonator.

    Recursive time-domain implementation of a resonator, matching Klatt's
    specification.

    Arguments:
        mast (KlattSynth): see parent class
        anti (boolean): determines whether Resonator acts as resonator or
            antiresonator (for more information, see Klatt 1980)

    Attributes:
        anti (boolean): See Arguments
    """
    def __init__(self, mast, anti=False):
        KlattComponent.__init__(self, mast)
        self.anti = anti

    def calc_coef(self, ff, bw):
        """
        Calculates filter coefficients.

        Calculates filter coefficients to implement resonator ala Klatt 1980.
        If self.anti = True, modifies the coefficients after calculation to
        turn the resonator into an antiresonator. Accepts center frequency and
        bandwidth values, and accesseds non-time-varying parameters from mast.

        Arguments:
            ff (array): Array of center frequency values in Hz, with length
                N_SAMP
            bw (array): Array of bandwidth values in Hz, with length N_SAMP
        """
        c = -np.exp(-2*np.pi*bw*self.mast.params["DT"])
        b = (2*np.exp(-np.pi*bw*self.mast.params["DT"])\
             *np.cos(2*np.pi*ff*self.mast.params["DT"]))
        a = 1-b-c
        if self.anti:
            a_prime = 1/a
            b_prime = -b/a
            c_prime = -c/a
            return(a_prime, b_prime, c_prime)
        else:
            return(a, b, c)

    def resonate(self, ff, bw):
        """
        Processes input waveform with resonator filter.

        Loops through values in the input array, calculating filter outputs
        sample-by-sample in the time domain. Takes arrays to indicate center
        frequency and bandwidth values, and passes them to calc_coef() to get
        coefficients to be used in the filtering calculation.

        Arguments:
            ff (array): Array of center frequency values in Hz, with length
                N_SAMP
            bw (array): Array of bandwidth values in Hz, with length N_SAMP
        """
        a, b, c = self.calc_coef(ff, bw)
        self.output[0] = a[0]*self.input[0]
        if self.anti:
            self.output[1] = a[1]*self.input[1] + b[1]*self.input[0]
            for n in range(2, self.mast.params["N_SAMP"]):
                self.output[n] = a[n]*self.input[n] + b[n]*self.input[n-1] \
                                + c[n]*self.input[n-2]
        else:
            self.output[1] = a[1]*self.input[1] + b[1]*self.output[0]
            for n in range(2,self.mast.params["N_SAMP"]):
                self.output[n] = a[n]*self.input[n] + b[n]*self.output[n-1] \
                                + c[n]*self.output[n-2]
        self.send()


class Impulse(KlattComponent):
    """
    Time-varying impulse generator.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        last_glot_pulse (int): Number of samples since last glottal pulse
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)
        self.last_glot_pulse = 0

    def impulse_gen(self, F0):
        """
        Generates impulse train.

        Starts with array of zeros with length N_SAMP. Loops through array,
        setting value to 1 when the time since last glottal pulse is equal
        to or exceeds the current glotal period (inverse of current F0).

        Arguments:
            F0 (arrry): Array of F0 values at each sample
        """
        glot_period = np.round(self.mast.params["FS"]/F0)
        self.last_glot_pulse = 0
        for n in range(self.mast.params["N_SAMP"]):
            if n - self.last_glot_pulse >= glot_period[n]:
                self.output[n] = 1
                self.last_glot_pulse = n
        self.send()


class Mixer(KlattComponent):
    """
    Mixes waveforms together.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def receive(self, signal):
        """
        Mixes incoming waveform with current input.

        Replaces KlattComponent's receive() method. Instead of setting input
        equal to incoming waveform, mixes input with incoming waveform.

        Arguments:
            signal (array): waveform to be mixed with input
        """
        self.input[:] = self.input[:] + signal[:]

    def mix(self):
        """
        Sets output to input.

        The above receive() method really does the mixing --- this is just the
        method called by the Mixer's KlattSection so that the signal
        propagates.
        """
        self.output[:] = self.input[:]
        self.send()


class Amplifier(KlattComponent):
    """
    Simple amplifier, scales amplitude of signal by dB value.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def amplify(self, dB):
        """
        Peforms amplification.

        Arguments:
            dB (float): amount of amplification to occur in dB
        """
        dB = np.sqrt(10)**(dB/10)
        self.output[:] = self.input[:]*dB
        self.send()


class Firstdiff(KlattComponent):
    """
    Simple first difference operator.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def differentiate(self):
        """
        Peforms first difference operation.
        """
        self.output[0] = 0
        for n in range(1, self.mast.params["N_SAMP"]):
            self.output[n] = self.input[n] - self.input[n-1]
        self.send()


class Lowpass(KlattComponent):
    """
    Simple one-zero 6 dB/oct lowpass filter.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def filter(self):
        """
        Implements lowpass filter operation.
        """
        self.output[0] = self.input[0]
        for n in range(1, self.mast.params["N_SAMP"]):
            self.output[n] = self.input[n] + self.output[n-1]
        self.send()


class Normalizer(KlattComponent):
    """
    Normalizes signal so that abs(max value) is 1.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def normalize(self):
        """
        Implements normalization.
        """
        self.output[:] = self.input[:]/np.max(np.abs(self.input[:]))
        self.send()


class Noisegen(KlattComponent):
    """
    Generates noise from a Gaussian distribution.

    Arguments:
        mast (KlattSynth): see parent class
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)

    def generate(self):
        """
        Generates Gaussian noise with mean 0, sd 1.0, and length N_SAMP
        """
        self.output[:] = np.random.normal(loc=0.0, scale=1.0,
                                          size=self.mast.params["N_SAMP"])
        self.send()


class Switch(KlattComponent):
    """
    Binary switch between two outputs.

    Has two output signals (instead of one, as in other KlattComponents). Each
    is connected to a different destination, and the operate() function
    switches the input between the two possible outputs depending on a control
    singal.

    Arguments:
        mast (KlattSynth): see parent class

    Attributes:
        output (list): List of two np.arrays as described above
    """
    def __init__(self, mast):
        KlattComponent.__init__(self, mast)
        self.output = []
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))

    def send(self):
        """
        Perpetuates signal to components further down in the chain.

        Replaces KlattComponent's send() method, sending one output to one
        destination and the other output to another destination.
        """
        self.dests[0].receive(signal=self.output[0][:])
        self.dests[1].receive(signal=self.output[1][:])

    def operate(self, choice):
        """
        Implements binary switching.

        Arguments:
            choice (np.array): Array of zeros and ones which tell the Switch
                where to send the input singal. For samples where switch=0 the
                signal is sent to the first output and the second output is set
                to zero. For samples where switch=1 the signal is sent to the
                second output and the first output is set to zero.
        """
        for n in range(self.mast.params["N_SAMP"]):
            if choice[n] == 0:
                self.output[0][n] = self.input[n]
                self.output[1][n] = 0
            elif choice[n] == 1:
                self.output[0][n] = 0
                self.output[1][n] = self.input[n]
        self.send()

    def clean(self):
        self.output = []
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))
        self.output.append(np.zeros(self.mast.params["N_SAMP"]))


if __name__ == '__main__':
    s = klatt_make(KlattParam1980(DUR=0.5)) # Creates a Klatt synthesizer w/ default settings
    # see also: http://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf
    N = s.params["N_SAMP"]
    F0 = s.params["F0"]
    FF = np.asarray(s.params["FF"]).T
    AV = s.params["AV"]
    AH = s.params['AH']

    # amplitude / voicing
    AV[:] = np.linspace(1, 0, N) ** 0.1 * 60
    if 1:  # unvoiced consonant
        Nv1 = 800  # start of unvoiced-voiced transition
        Nv2 = 1000  # end of unvoiced-voiced transition
        AV[:Nv1] = 0
        AH[:Nv1] = 55
        AV[Nv1:Nv2] = np.linspace(0, AV[Nv2], Nv2-Nv1)
        AH[Nv1:Nv2] = np.linspace(55, 0, Nv2-Nv1)


    # F0
    F0[:] = np.linspace(120, 70, N)  # a falling F0 contour

    # FF
    target1 = np.r_[300, 1000, 2600]  # /b/
    #target2 = np.r_[280, 2250, 2750]  # /i/
    target2 = np.r_[750, 1300, 2600]  # /A/
    if 0:  # linear transition
        xfade = np.linspace(1, 0, N)
    else:  # exponential transition
        n = np.arange(N)
        scaler = 20
        xfade = 2 / (1 + np.exp(scaler * n / (N-1)))
    FF[:,:3] = np.outer(xfade, target1) + np.outer((1 - xfade), target2)

    # synthesize
    s.params["FF"] = FF.T
    s.run()
    s.play()
    s.save('synth.wav')

    # visualize
    t = np.arange(len(s.output)) / s.params['FS']
    import matplotlib.pyplot as plt
    ax = plt.subplot(211)
    plt.plot(t, s.output)
    plt.axis(ymin=-1, ymax=1)
    plt.ylabel('amplitude')
    plt.twinx()
    plt.plot(t, AV, 'r', label='AV')
    plt.plot(t, AH, 'g', label='AH')
    plt.legend()
    plt.subplot(212, sharex=ax)
    plt.specgram(s.output, Fs=s.params['FS'])
    plt.plot(t, FF, alpha=0.5)
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.savefig('figure.pdf')
    plt.show()
