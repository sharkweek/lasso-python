
import glob
from typing import List, Union

import h5py
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly import colors

from .lsda_py3 import Lsda

'''
# Recoded stuff from lsda from LSTC, but much more readable and quoted ...
#
class Diskfile:

    # symbol = binary variable / python translation
    # note: bolded means unsigned
    #
    # b = char / int
    # h = short / int
    # i = int   / int
    # q = long long / int
    # f = float  / float
    # d = double / float
    # s = char[] / string

    packsize = [0,"b","h",0,"i",0,0,0,"q"]
    packtype = [0,"b","h","i","q","B","H","I","Q","f","d","s"]
    sizeof = [0,1,2,4,8,1,2,4,8,4,8,1] # packtype

    ##
    #
    #
    def __init__(self,filepath,mode="r"):

        # This opens a file and mainly treats the header
        #
        # header[0] == header length & header offset ?!? | default=8 byte
        # header[1] == lengthsize  | default = 8 byte
        # header[2] == offsetsize  | default = 8 byte
        # header[3] == commandsize | default = 1 byte
        # header[4] == typesize    | default = 1 byte
        # header[5] == file endian | default = 1 byte
        # header[6] == ?           | default = 0 byte
        # header[7] == ?           | default = 0 byte

        # start init
        self.filepath = filepath
        self.mode = mode
        self.file_ends = False

        # open file ...
        self.fp = open(filepath,mode+"b")
        # ... in read mode yay
        if mode == "r":
            header = struct.unpack("BBBBBBBB",self.fp.read(8))
            if header[0] > 8: #?!? some kind of header offset ?!?
                self.fp.seek(header[0])
        # ... in write mode
        else:
            header = [8,8,8,1,1,0,0,0]
            if sys.byteorder == "big":
                header[5] = 0
            else:
                header[5] = 1

        # fetch byte length of several ... I honestly don't know what exactly
        self.lengthsize    = header[1]
        self.offsetsize    = header[2]
        self.commandsize   = header[3]
        self.typesize      = header[4]
        self.ordercode = ">" if header[5] == 0 else '<' # endian

        # again I have no idea what is going on ...
        # these are some data unpacking format codes
        self.ounpack  =  self.ordercode+Diskfile.packsize[self.offsetsize]
        self.lunpack  =  self.ordercode+Diskfile.packsize[self.lengthsize]
        self.lcunpack = (self.ordercode+
                        Diskfile.packsize[self.lengthsize]+
                        Diskfile.packsize[self.commandsize])
        self.tolunpack = (self.ordercode+
                        Diskfile.packsize[self.typesize]+
                        Diskfile.packsize[self.offsetsize]+
                        Diskfile.packsize[self.lengthsize])
        self.comp1 = self.typesize+self.offsetsize+self.lengthsize
        self.comp2 = self.lengthsize+self.commandsize+self.typesize+1

        # write header if write mode
        if mode == "w":
            header_str = ''
            for value in header:
                s += struct.pack("B",value) # convert to unsigned char
            self.fp.write(s)
            # self.writecommand(17,Lsda.SYMBOLTABLEOFFSET)
            # self.writeoffset(17,0)
            # self.lastoffset = 17

    # UNFINISHED
'''


class Binout:
    '''This class is meant to read binouts from LS-Dyna

    Parameters
    ----------
    filepath: str
        path to the binout

    Notes
    -----
        This class is only a utility wrapper for Lsda from LSTC.

    Examples
    --------
        >>> binout = Binout("path/to/binout")
    '''

    default_plot_layout = go.Layout({
        'title': {'x': 0.5},
        'xaxis': {'tickformat': '.3s'},
        'yaxis': {'tickformat': '.3s'},
        'font': {'family': 'Segoe UI',
                 'size': 14},
        'template': 'plotly_white',
        'colorway': colors.qualitative.Plotly,
        'hovermode': 'x',
        'width': 800,
        'height': 500
    })
    
    def __init__(self, filepath: str):
        '''Constructor for a binout

        Parameters
        ----------
        filepath: str
            path to the binout or pattern

        Notes
        -----
            The class loads the file given in the filepath. By giving a
            search pattern such as: "binout*", all files with that
            pattern will be loaded.

        Examples
        --------
            >>> # reads a single binout
            >>> binout = Binout("path/to/binout0000")
            >>> binout.filelist
            ['path/to/binout0000']

            >>> # reads multiple files
            >>> binout = Binout("path/to/binout*")
            >>> binout.filelist
            ['path/to/binout0000','path/to/binout0001']
        '''

        self.filelist = glob.glob(filepath)

        # check file existance
        if not self.filelist:
            raise IOError("No file was found.")

        # open lsda buffer
        self.lsda = Lsda(self.filelist, "r")
        self.lsda_root = self.lsda.root

        # if sys.version_info[0] < 3:
        #    self.lsda_root = self.lsda.root
        # else:
        #    self.lsda_root = self.lsda.root.children[""]
        # self.lsda_root = self.lsda.root

    def read(self, *path) -> Union[List[str], str, np.ndarray]:
        '''Read all data from Binout (top to low level)

        Parameters
        ----------
        path: Union[Tuple[str, ...], List[str], str]
            internal path in the folder structure of the binout

        Returns
        -------
        ret: Union[List[str], str, np.ndarray]
            list of subdata within the folder or data itself (array or string)

        Notes
        -----
            This function is used to read any data from the binout. It has been used
            to make the access to the data more comfortable. The return type depends
            on the given path:

             - `binout.read()`: `List[str] names of directories (in binout)
             - `binout.read(dir)`: `List[str]` names of variables or subdirs
             - `binout.read(dir1, ..., variable)`: np.array data

            If you have multiple outputs with different ids (e.g. in nodout for
            multiple nodes) then don't forget to read the id array for
            identification or id-labels.

        Examples
        --------
            >>> from lasso.dyna import Binout
            >>> binout = Binout("test/binout")
            >>> # get top dirs
            >>> binout.read()
            ['swforc']
            >>> binout.read("swforc")
            ['title', 'failure', 'ids', 'failure_time', ...]
            >>> binout.read("swforc","shear").shape
            (321L, 26L)
            >>> binout.read("swforc","ids").shape
            (26L,)
            >>> binout.read("swforc","ids")
            array([52890, 52891, 52892, ...])
            >>> # read a string value
            >>> binout.read("swforc","date")
            '11/05/2013'
        '''

        return self._decode_path(path)

    def legend(self, db: str) -> pd.DataFrame:
        """Legend as a pandas.DataFrame

        Parameters
        ----------
        db : str
            The database for the desired legend

        Returns
        -------
        pandas.DataFrame
            Legend with ID and title pairs

        Raises
        ------
        ValueError
            if specified database does not have a legend

        Example
        -------
        >>> from lasso.dyna import Binout
        >>> binout = Binout('path/to/binout')
        >>> binout.legend('matsum')

              id     title
        0      1    Part 1
        1      2    Part 2
        2      3    Part 3
        3      4    Part 4
        4      5    Part 5
        ...  ...       ...
        94    95   Part 95
        95    96   Part 96
        96    97   Part 97
        97    98   Part 98
        98    99   Part 99
        99   100  Part 100
        """
    
        # validate legend exists
        if 'legend' not in self.read(db):
            raise ValueError(db + " has no legend")

        legend = self.read(db, 'legend')

        if 'legend_ids' in self.read(db):
            id_field = 'legend_ids'
        else:
            id_field = 'ids'

        # parse string into dataframe
        return pd.DataFrame({
            'id': self.read(db, id_field),
            'title': [legend[i:i + 80].strip()
                      for i in range(0, len(legend), 80)]
        })

    def as_df(self, *args) -> pd.DataFrame:
        """ read data and convert to pandas dataframe if possible 
        
        Parameters
        ----------
        path: Union[Tuple[str, ...], List[str], str]
            internal path in the folder structure of the binout
        
        Returns
        -------
        df: pandas.DataFrame
            data converted to pandas dataframe
        
        Raises
        ------
        ValueError
            if the data cannot be converted to a pandas dataframe
        
        Examples
        --------
            >>> from lasso.dyna import Binout
            >>> binout = Binout('path/to/binout')
            
            Read a time-dependent array.
            
            >>> binout.as_df('glstat', 'eroded_kinetic_energy')
            time
            0.00000        0.000000
            0.19971        0.000000
            0.39942        0.000000
            0.59976        0.000000
            0.79947        0.000000
                            ...
            119.19978    105.220786
            119.39949    105.220786
            119.59983    105.220786
            119.79954    105.220786
            119.99988    105.220786
            Name: eroded_kinetic_energy, Length: 601, dtype: float64

            Read a time and id-dependent array.

            >>> binout.as_df('secforc', 'x_force')
                                  1             2             3  ...            33            34            35
            time                                                 ...
            0.00063    2.168547e-16  2.275245e-15 -3.118639e-14  ... -5.126108e-13  4.592941e-16  8.431434e-17
            0.20034    3.514243e-04  3.797908e-04 -1.701294e-03  ...  2.530416e-11  2.755493e-07  2.117375e-05
            0.40005    3.052490e-03  3.242951e-02 -2.699926e-02  ...  6.755315e-06 -2.608923e-03  3.919351e-03
            0.60039   -1.299816e-02  4.930999e-02 -1.632376e-02  ...  8.941705e-05 -2.203455e-02  3.536490e-02
            0.80010    1.178485e-02  4.904512e-02 -9.740204e-03  ...  5.648263e-05 -6.999854e-02  6.934055e-02
            ...                 ...           ...           ...  ...           ...           ...           ...
            119.00007  9.737679e-01 -8.833702e+00  1.298964e+01  ... -9.977377e-02  7.883521e+00 -5.353501e+00
            119.20041  7.421170e-01 -8.849411e+00  1.253505e+01  ... -1.845916e-01  7.791409e+00 -4.988928e+00
            119.40012  9.946615e-01 -8.541475e+00  1.188757e+01  ... -3.662228e-02  7.675800e+00 -4.889339e+00
            119.60046  9.677638e-01 -8.566695e+00  1.130774e+01  ...  5.144208e-02  7.273052e+00 -5.142375e+00
            119.80017  1.035165e+00 -8.040828e+00  1.124044e+01  ... -1.213450e-02  7.188395e+00 -5.279221e+00
        """

        data = self.read(*args)

        # validate time-based data
        if not isinstance(data, np.ndarray):
            err_msg = "data is not a numpy array but has type '{0}'"
            raise ValueError(err_msg.format(type(data)))

        time_array = self.read(*args[:-1], 'time')
        if data.shape[0] != time_array.shape[0]:
            raise ValueError(
                "data pd.Series length does not match time array length"
            )

        time_pdi = pd.Index(time_array, name='time')

        # create dataframe
        if data.ndim > 1:
            df = pd.DataFrame(index=time_pdi)

            # get names for rcforc columns
            if args[0] == 'rcforc':
                ids = [(str(i) + 'm') if j else (str(i) + 's')
                       for i, j in zip(self.read('rcforc', 'ids'),
                                       self.read('rcforc', 'side'))]
            # get names by id
            elif ('ids' in self.read(*args[:-1])
                  and self.read(*args[:-1], 'ids').shape[0]
                  == self.read(*args).shape[1]):
                # all other column names
                ids = self.read(*args[:-1], 'ids')
            # get names by legend
            elif ('legend_ids' in self.read(*args[:-1])
                  and self.read(*args[:-1], 'legend_ids').shape[0]
                  == self.read(*args).shape[1]):
                ids = self.read(*args[:-1], 'legend_ids')
            else:
                ids = None

            if ids is not None:
                for i, j in enumerate(ids):
                    df[str(j)] = data.T[i]
            else:  # create titles for nameless tensors
                for i, j in enumerate(data.T):
                    if args[-1][-1] == 's':
                        col_name = '-'.join([args[-1][:-1], str(i + 1)])
                    else:
                        col_name = '-'.join([args[-1], str(i + 1)])

                    df[col_name] = j

        else:
            df = pd.DataFrame({args[-1]: data},
                              index=time_pdi)

        return df

    # udpate other functions that use new format before replacing existing as_df
    def _as_df(self, db: str,
               include: list = [],
               exclude: list = []) -> pd.DataFrame:
        """Cast time-based binout data as a pandas DataFrame

        Parameters
        ----------
        db : str
            binout database (e.g. 'glstat', 'matsum', etc.)
        include : list, optional
            data search strings to include from DataFrame, by default []
        exclude : list, optional
            data search strings to exclude from results DataFrame,
            by default []

        Returns
        -------
        pandas.DataFrame
            DataFrame containing binout data indexed by time

        Raises
        ------
        ValueError
            if `db` is not in binout
        TypeError
            if `include` or `exclude` are not lists
        """

        # type check inputs
        if db not in self.read():
            raise ValueError('`db` not in binout')
        if not isinstance(include, list):
            raise TypeError("`include` must be a list")
        if not isinstance(exclude, list):
            raise TypeError("`exclude` must be a list")

        time = self.read(db, 'time')
        df = pd.DataFrame(index=time)
        tbd = [i for i in self.time_based_data(db) if i != 'time']
        out_col = list(set(tbd))  # known issue: duplicate 'cycle' in `tbd`

        # add data to dataframe
        for i in tbd:
            data = self.read(db, i)

            # nested data arrays (e.g. tensors)
            if len(data.shape) > 1:
                col = [i + '_' + str(j) for j in range(data.shape[1])]
                right_df = pd.DataFrame(data, columns=col, index=time)
                out_col.remove(i)
                out_col += col

            # single data arrays
            else:
                right_df = pd.DataFrame({i: data}, index=time)
                out_col.append(i)

            df = df.join(right_df)

        out_col = list(set(out_col))  # remove duplicates

        # filter out for include and exclude
        if include:
            out_col = [u for u in out_col for v in include if v in u]
        if exclude:
            out_col = [u for u in out_col for v in exclude if v not in u]

        return df[out_col]
    
    
    def time_based_data(self, db: str) -> list:
        time_fields = []
        time_shape = self.read(db, 'time').shape[0]

        for field in self.read(db):
            try:
                shp = (self.read(db, field).shape[0] == time_shape)
            except AttributeError:
                shp = False

            if shp and field != 'time':
                time_fields.append(field)

        return time_fields

    def plot(self, *args, **kwargs) -> go.Figure:
        special_func = {
            'matsum': self._plot_matsum
        }

        return special_func.get(args[0], self._plot_db)(*args, **kwargs)

    def _plot_db(self, db, include=None, exclude=None):
        time_fields = self.time_based_data(db)

        # type check include
        if include is not None:
            if type(include) == str:
                include = list((include,))
            elif type(include) != list:
                raise TypeError("`include` must be str or list of str")

            # filter
            data_fields = [i for i in time_fields for j in include if j in i]
        else:
            data_fields = time_fields

        if exclude is not None:
            # type check exclude
            if type(exclude) == str:
                exclude = list((exclude,))
            elif type(exclude) != list:
                raise TypeError("`exclude` must be str or list of str")

            data_fields = [
                i for i in data_fields for j in exclude if j not in i
            ]

        if len(data_fields) < 1:
            raise ValueError("No resulting data from include/exclude criteria")

        plot_df = pd.DataFrame(columns=['data_type', 'traces'])
        for i in data_fields:
            df = pd.DataFrame(self.as_df(db, i))
            traces = [go.Scatter(x=df.index, y=df[j], name=j) for j in df]

            plot_df = plot_df.append({
                'data_type': i,
                'traces': traces
            }, ignore_index=True)

        fig = go.Figure(layout=self.default_plot_layout)
        for i in plot_df.traces:
            fig.add_traces(i)

        return fig

    def _plot_matsum(self, db, part_filter=None, include=None,
                     exclude=None) -> go.Figure:
        time_fields = self.time_based_data(db)

        # type check include
        if include is not None:
            if type(include) == str:
                include = list((include,))
            elif type(include) != list:
                raise TypeError("`include` must be str or list of str")

            # filter
            data_types = [i for i in time_fields for j in include if j in i]
        else:
            data_types = time_fields

        if exclude is not None:
            # type check exclude
            if type(exclude) == str:
                exclude = list((exclude,))
            elif type(exclude) != list:
                raise TypeError("`exclude` must be str or list of str")

            data_types = [
                i for i in data_types for j in exclude if j not in i
            ]

        data = {}
        legend = self.legend(db)
        titles = {str(i): str(i) + ": " + j
                  for i, j in zip(legend.id, legend.title)}

        for each in data_types:
            if part_filter is not None:
                data[each] = self.as_df(db, each)[part_filter].rename(
                    columns=titles
                )
            else:
                data[each] = self.as_df(db, each).rename(
                    columns=titles
                )

        # create dataframe of all traces
        plot_df = pd.DataFrame(columns=['data_type', 'part', 'trace'])
        for data_type in data:
            for part in data[data_type]:
                plot_df = plot_df.append({
                    'data_type': data_type,
                    'part': part,
                    'trace': go.Scatter(
                        x=data[data_type].index,
                        y=data[data_type][part],
                        name=data_type,
                        visible=(part == data[data_type].columns[0])
                    )
                }, ignore_index=True)

        # create figure and add traces
        fig = go.Figure(layout=self.default_plot_layout)
        for trace in plot_df.trace:
            fig.add_trace(trace)

        # update figure with dropdown by data type
        dropdown = [{'label': i,
                     'method': 'update',
                     'args': [{'visible': plot_df['part'] == i},
                              {'yaxis': {'title': i}}]}
                    for i in plot_df['part'].unique()]

        fig.update_layout({'title': {'text': '`matsum` Data'},
                           'legend': {'title': 'Part'},
                           'hovermode': 'closest',
                           'yaxis': {'title': data_types[0]}},
                          updatemenus=[{'active': 0,
                                        'type': 'dropdown',
                                        'buttons': dropdown,
                                        'direction': "down",
                                        'x': 0.01,
                                        'xanchor': 'left',
                                        'y': 0.99,
                                        'yanchor': 'top',
                                        'showactive': True}])

        return fig
    
    def _decode_path(self, path):
        '''Decode a path and get whatever is inside.

        Parameters
        ----------
        path: List[str]
            path within the binout

        Notes
        -----
            Usually returns the folder children. If there are variables in the folder
            (usually also if a subfolder metadata exists), then the variables will
            be printed from these directories.

        Returns
        -------
        ret: Union[List[str], np.ndarray]
            either subfolder list or data array
        '''

        iLevel = len(path)

        if iLevel == 0:  # root subfolders
            return self._bstr_to_str(list(self.lsda_root.children.keys()))

        # some subdir
        else:

            # try if path can be resolved (then it's a dir)
            # in this case print the subfolders or subvars
            try:

                dir_symbol = self._get_symbol(self.lsda_root, path)

                if 'metadata' in dir_symbol.children:
                    return self._collect_variables(dir_symbol)
                else:
                    return self._bstr_to_str(list(dir_symbol.children.keys()))

            # an error is risen, if the path is not resolvable
            # this could be, because we want to read a var
            except ValueError as err:

                return self._get_variable(path)

    def _get_symbol(self, symbol, path):
        '''Get a symbol from a path via lsda

        Parameters
        ----------
        symbol: Symbol
            current directory which is a Lsda.Symbol

        Returns
        -------
        symbol: Symbol
            final symbol after recursive search of path
        '''

        # check
        if symbol == None:
            raise ValueError("Symbol may not be none.")

        # no further path, return current symbol
        if len(path) == 0:
            return symbol
        # more subsymbols to search for
        else:

            sub_path = list(path)  # copy
            next_symbol_name = sub_path.pop(0)

            next_symbol = symbol.get(next_symbol_name)
            if next_symbol == None:
                raise ValueError("Cannot find: %s" % next_symbol_name)

            return self._get_symbol(next_symbol, sub_path)

    def _get_variable(self, path):
        '''Read a variable from a given path

        Parameters
        ----------
        path: List[str]
            path to the variable

        Returns
        -------
        data: np.ndarray
        '''

        dir_symbol = self._get_symbol(self.lsda_root, path[:-1])
        # variables are somehow binary strings ... dirs not
        variable_name = self._str_to_bstr(path[-1])

        # var in metadata
        if ("metadata" in dir_symbol.children) and (variable_name in dir_symbol.get("metadata").children):
            var_symbol = dir_symbol.get("metadata").get(variable_name)
            var_type = var_symbol.type

            # symbol is a string
            if var_type == 1:
                return self._to_string(var_symbol.read())
            # symbol is numeric data
            else:
                return np.asarray(var_symbol.read())

        # var in state data ... hopefully
        else:

            time = []
            data = []
            for subdir_name, subdir_symbol in dir_symbol.children.items():

                # skip metadata
                if subdir_name == "metadata":
                    continue

                # read data
                if variable_name in subdir_symbol.children:
                    state_data = subdir_symbol.get(variable_name).read()
                    if len(state_data) == 1:
                        data.append(state_data[0])
                    else:  # more than one data entry
                        data.append(state_data)

                    time_symbol = subdir_symbol.get(b"time")
                    if time_symbol:
                        time += time_symbol.read()
                    # data += subdir_symbol.get(variable_name).read()

            # return sorted by time
            if len(time) == len(data):
                return np.array(data)[np.argsort(time)]
            else:
                return np.array(data)

    def _collect_variables(self, symbol):
        '''Collect all variables from a symbol

        Parameters
        ----------
        symbol: Symbol

        Returns
        -------
        variable_names: List[str]

        Notes
        -----
            This function collect all variables from the state dirs and metadata.
        '''

        var_names = set()
        for _, subdir_symbol in symbol.children.items():
            var_names = var_names.union(subdir_symbol.children.keys())

        return self._bstr_to_str(list(var_names))

    def _to_string(self, data_array):
        '''Convert a data series of numbers (usually ints) to a string

        Parameters
        ----------
        data: Union[int, np.ndarray]
            some data array

        Returns
        -------
        string: str
            data array converted to characters

        Notes
        -----
            This is needed for the reason that sometimes the binary data
            within the files are strings.
        '''

        return "".join([chr(entry) for entry in data_array])

    def _bstr_to_str(self, arg):
        '''Encodes or decodes a string correctly regarding python version

        Parameters
        ----------
        string: Union[str, bytes]

        Returns
        -------
        string: str
            converted to python version
        '''

        # in case of a list call this function with its atomic strings
        if isinstance(arg, (list, tuple)):
            return [self._bstr_to_str(entry) for entry in arg]

        # convert a string (dependent on python version)
        if not isinstance(arg, str):
            return arg.decode("utf-8")
        else:
            return arg

    def _str_to_bstr(self, string):
        '''Convert a string to a binary string python version independent

        Parameters
        ----------
        string: str

        Returns
        -------
        string: bytes
        '''

        if not isinstance(string, bytes):
            return string.encode("utf-8")
        else:
            return string

    def save_hdf5(self, filepath, compression="gzip"):
        ''' Save a binout as HDF5

        Parameters
        ----------
        filepath: str
            path where the HDF5 shall be saved
        compression: str
            compression technique (see h5py docs)

        Examples
        --------
            >>> binout = Binout("path/to/binout")
            >>> binout.save_hdf5("path/to/binout.h5")
        '''

        with h5py.File(filepath, "w") as fh:
            self._save_all_variables(fh, compression)

    def _save_all_variables(self, hdf5_grp, compression, *path):
        ''' Iterates through all variables in the Binout

        Parameters
        ----------
        hdf5_grp: Group
            group object in the HDF5, where all the data
            shall be saved into (of course in a tree like
            manner) 
        compression: str
            compression technique (see h5py docs)
        path: Tuple[str, ...]
            entry path in the binout
        '''

        ret = self.read(*path)
        path_str = "/".join(path)

        # iterate through subdirs
        if isinstance(ret, list):

            if path_str:
                hdf5_grp = hdf5_grp.create_group(path_str)

            for entry in ret:
                path_child = path + (entry,)
                self._save_all_variables(
                    hdf5_grp, compression, *path_child)
        # children are variables
        else:
            # can not save strings, only list of strings ...
            if isinstance(ret, str):
                ret = np.array([ret], dtype=np.dtype("S"))
            hdf5_grp.create_dataset(
                path[-1], data=ret, compression=compression)
