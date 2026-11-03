from astropy.time import Time
import astropy.units as u
import numpy as np
from getStationConfig import GetStationConfig
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
import sys
import matplotlib.pyplot as plt
import subprocess
import datetime
import shlex
import re
import os

def time_to_at(t):
    """
    Convert an astropy.Time object into the string time date format used by at on linux
    Parameters
    ----------
    t : astropy.Time
        Desired time
    Returns
    -------
    timeStr : string
        The year-month-day hr:min formated time string for at
    seconds : float
        The residual time between the timeStr and the input time
    """
    dt = t.datetime
    seconds = dt.second+dt.microsecond/1e6
    dt+=datetime.timedelta(seconds=-seconds)
    timeStr = f'{dt.time().isoformat()[:5]} {dt.date().isoformat()}'
    return(timeStr,seconds)

def str_to_time(t):
    """
    Parameters
    ----------
    Returns
    -------
    """
    return Time(f"{t[:4]}-{t[4:6]}-{t[6:8]}T{t[9:11]}:{t[11:13]}:{t[13:]}")

def alaz_to_thetas(alt,az,stationAngle = 3*u.deg):
    """
    Determine the East-West and North-South angles for a given Altitude and Azimuth
    Parameters
    ----------
    alt : astropy.Quantity
        Altitute of target.
    az : String (optional)
        Azimuth of target.
    stationAngle : astropy.Quantity (optional)
        Angle offset of station NS from true NS. Defaults to 3 deg (FUS value)
    Returns
    -------
    thEW : astropy.Quantity
        Angular position of source in beamform East-West direction
    thNS : astropy.Quantity
        Angular position of source in beamform North-South direction
    """
    ## convert into cartesian coordinates
    ## correct Az for rotation of the array relative to NS
    x = np.cos(az+stationAngle)
    y = np.sin(az+stationAngle)
    z = np.sin(alt)
    x *= np.cos(alt)
    y *=np.cos(alt)
    hx = np.sqrt(x**2+z**2)
    hy = np.sqrt(y**2+z**2)
    ## Convert into the two angles
    thNS = np.arctan2(x,hy).to(u.deg)
    thEW = np.arctan2(y,hx).to(u.deg)
    return(thEW,thNS)

def time_to_string(time):
    dt = time.datetime
    string = str(dt)
    string = string.replace("-", "")
    string = string.replace(" ", "_")
    string = string.replace(":", "")
    return string

def get_disks(machine,username='ubuntu'):
    """
    Returns the free space available on a given BURSTT machine
    Parameters
    ----------
    machine: string
        The name of BURSTT machine to check (e.g. burstt11)
    username: string (optional)
        The username to use when connecting to the machine. Defaults to ubuntu
    Returns
    -------
    diskNums: np.ndarray
        List of disk numbers
    space: np.ndArray
        The space available within each disk (in Tibytes)
    """
    result = subprocess.run([f'ssh {username}@{machine}  \'df -lh | grep "dev/s" | grep "disk"\''],
                                text=True,
                                capture_output=True,
                                shell=True).stdout
    drivePars = [[y for y in x.split(' ') if len(y)>0] for x in result.split('\n') if len(x)>0]
    diskNums = np.array([int(pars[-1].split('disk')[-1]) for pars in drivePars])
    availStr = [pars[3] for pars in drivePars]
    space = np.array([float(x[:-1])*getattr(u,x[-1]+'ibyte').to(u.Tibyte) for x in availStr])*u.Tibyte
    return(diskNums,space)

class ScheduleItem:
    """
    Container to hold information about an item in the BURSTT schedule
    """
    def __init__(self,note: str ='',stations: list[str] = [], **kwargs):
        self.note = note
        self.stations = stations
        self.status = 'planning'
        pass

    def from_dict(self,dict: dict):
        """
        Parameters
        ----------
        dict: dictionary
            Dictionary with schedule Item parameters
        """
        if 'note' in dict:
            self.note = dict['note']
        else:
            self.note = ''
        self.stations = dict['stations']
        self.status = dict['status']

    def check_close(self,time: Time, tol=0*u.hour):
        """
        Test if a given time is close to this event
        Parameters
        ----------
        time : astropy.Time

        Returns
        -------
            False
        """
        return(False)

    def to_dict(self) -> dict:
        """
        Put standard parameters in a dictionary for saving
        Returns
        -------
        pars : dictionary
            A dictionary of the relevant parameters
        """
        pars = {'note':self.note,
                'stations':self.stations,
                'status':self.status
               }
        return(pars)
    
    def __repr__(self):
        return f'ScheduleItem({self.status})'

    def plot_kwargs(self) -> dict:
        """
        Determine the kwargs needed to add the event to the scheduling plot
        Returns
        -------
        pars: dict
            A dictionary of plotting options
        """
        pars = {'alpha':.9}
        return(pars)

    @property
    def id(self):
        return('')

    @property
    def info(self):
        label = 'Scheduled Event\n'
        label += f'Status: {self.status}\n'
        label += f'Stations: {self.stations}\n'
        label += f'{self.note}'
        return(label)
    

class ScheduleInstant(ScheduleItem):
    """
    Containter to hold information about an instantaneous scheduled event
    """
    def __init__(self,time: Time = Time(0,format='unix'), **kwargs):
        self.time = time
        super().__init__(**kwargs)

    def from_dict(self,dict: dict):
        super().from_dict(dict)
        self.time = Time(dict['time'])

    def check_close(self,time: Time, tol=0*u.hour):
        return(np.abs(self.time-time).to(u.hour)<tol)

    def to_dict(self):
        dict = super().to_dict()
        dict.update({'time' : self.time.isot})
        return(dict)   

    def __lt__(self,other) -> bool:
        if isinstance(other,ScheduleInstant):
            return self.time<other.time
        if isinstance(other,ScheduleExtended):
            return self.time<other.start

    def __repr__(self):
        return f'ScheduleInstant({self.time.isot})'

    def plot_kwargs(self) -> dict:
        plotOptions = super().plot_kwargs()
        plotOptions.update({'linewidth':5})
        if self.status in ['submitted','bemformed']:
            plotOptions.update({'ls':'-'})
        elif self.status == 'pending':
            plotOptions.update({'ls':'--'})
        else:
            plotOptions.update({'ls':':'})
        return plotOptions

    def check_overlap(self,other) -> bool:
        if isinstance(other,ScheduleInstant):
            return self.time==other.time
        if isinstance(other,ScheduleExtended):
            return (self.time>=other.start)*(self.time<=other.end)
        
    @property
    def id(self):
        return((self.time-8*u.hour).isot.replace('-','').replace('T','_').replace(':','').split('.')[0])
    @property
    def info(self):
        label = 'Scheduled Event\n'
        label += f'Run at: {self.time.isot}\n'
        label += f'Status: {self.status}\n'
        label += f'Stations: {self.stations}\n'
        label += f'{self.note}'
        return(label)

class ScheduleExtended(ScheduleItem):
    """
    Containter to hold information about an ongoing scheduled event
    """
    def __init__(self,start: Time = Time(0,format='unix'), end: Time = Time(1,format='unix'),**kwargs):
        self.start: Time = start
        self.end: Time = end
        super().__init__(**kwargs)

    def from_dict(self,dict: dict):
        super().from_dict(dict)
        self.start = Time(dict['start'])
        self.end = Time(dict['end'])

    def to_dict(self):
        dict = super().to_dict()
        dict.update({'start' : self.start.isot,
                     'end' : self.end.isot
                     })
        return(dict)  

    def check_close(self,time: Time,tol=0*u.hour):
        return((time>self.start)*(time<self.end))

    def __lt__(self,other):
            if isinstance(other,ScheduleInstant):
                return self.start<other.time
            if isinstance(other,ScheduleExtended):
                return self.start<other.start
    def __repr__(self):
        return f'ScheduleExtended({self.start.isot})'

    def plot_kwargs(self) -> dict:
            plotOptions = super().plot_kwargs()
            if self.status == 'pending':
                plotOptions.update({'hatch':'///','facecolor':'none'})
            elif self.status == 'planning':
                plotOptions.update({'hatch':"\\\\",'facecolor':'none'})
            return plotOptions

    def check_overlap(self,other) -> bool:
        if isinstance(other,ScheduleInstant):
            return (other.time>=self.start)*(other.time<=self.end)
        if isinstance(other,ScheduleExtended):
            return(not ((self.end<other.start)+(self.start>other.end)))
    
    @property
    def id(self):
        return((self.start-8*u.hour).isot.replace('-','').replace('T','_').replace(':','').split('.')[0])
    @property
    def info(self):
        label = 'Scheduled Event\n'
        label += f'Run from {self.start.isot} to {self.stop.isot}\n'
        label += f'Status: {self.status}\n'
        label += f'Stations: {self.stations}\n'
        label += f'{self.note}'
        return()

class ScheduleBeam(ScheduleExtended):
    """
    Containter to hold information about a single beam recording event
    """
    def __init__(self,target: str = '', **kwargs):
        self.target = target
        self.drives = {}
        self.files = {}
        self.beamParameters = {}
        self.servers = {}
        super().__init__(**kwargs)

    def from_dict(self,dict: dict):
        super().from_dict(dict)
        self.target = dict['target']
        self.drives = dict['drives']
        self.files = dict['files']
        self.beamParameters = dict['beamParameters']
        if 'servers' in dict:
            self.servers = dict['servers']

    def to_dict(self):
        dict = super().to_dict()
        dict.update({'target' : self.target})
        dict.update({'drives' : self.drives})
        dict.update({'files' : self.files})
        dict.update({'beamParameters' : self.beamParameters})
        dict.update({'servers': self.servers})
        return(dict)

    def __repr__(self):
        return f'ScheduleBeam({self.start.isot},{self.target})'

    def plot_kwargs(self) -> dict:
            plotOptions = super().plot_kwargs()
            if self.status in ['submitted','transferred']:
                plotOptions.update({'facecolor':'tab:blue'})
            else:
                plotOptions.update({'edgecolor':'tab:blue'})
            return plotOptions
    
    def strip_files(self) -> None:
        for station in self.stations:
            with open(self.files[station],'r') as file:
                lines = file.readlines()
            newLines = []
            for line in lines:
                if (line[0] == '#') or ('transit' in line):
                    newLines.append(line)
                    continue
                ts = str_to_time(line.split(' ')[0])
                te = str_to_time(line.split(' ')[1])
                if self.start>= te:
                    continue
                if self.end <=ts:
                    continue
                newLines.append(line)
            newLines[-1] = newLines[-1].removesuffix('\n')
            newFile = self.files[station].removesuffix('.txt')
            newFile += f'{self.id}.txt'
            self.files[station] = newFile
            with open(newFile,'w') as file:
                file.writelines(newLines)

    @property
    def info(self):
        label = 'Single Beam Event\n'
        label += f'Run from {self.start.isot} to {self.end.isot}\n'
        label += f'Target: {self.target}\n'
        label += f'Status: {self.status}\n'
        label += f'Stations: {self.stations}\n'
        label += f'{self.note}\n'
        return(label)

class SchedulePoint(ScheduleInstant):
    """
    Containter to hold information about a scheduled repointing event
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.updates = {station : {} for station in self.stations}
        self.delays = {station : {} for station in self.stations}

    def from_dict(self,dict: dict):
            super().from_dict(dict)
            if 'updates' in dict:
                self.updates = dict['updates']
            if 'delays' in dict:
                self.delays = dict['delays']
    def to_dict(self):
        dict = super().to_dict()
        dict.update({'updates' : self.updates})
        dict.update({'delays' : self.delays})
        return(dict)
    
    def __repr__(self):
        return f'SchedulePoint({self.time.isot})'

    def plot_kwargs(self) -> dict:
        plotOptions = super().plot_kwargs()
        plotOptions.update({'color':'tab:orange'})
        return(plotOptions)

    @property
    def info(self):
        label = 'Repointing\n'
        label += f'Run at: {self.time.isot}\n'
        label += f'Status: {self.status}\n'
        label += f'Stations:\n'
        for station in self.stations:
            if station in self.updates:
                label+=f'{station}\n'
                for parameter in self.updates[station]:
                    label+=f'-{parameter}:{self.updates[station][parameter]}\n'
        label += f'{self.note}'
        return(label)
    

class ScheduleTrigger(ScheduleInstant):
    """
    Containter to hold information about a scheduled trigger event
    """
    def __init__(self,target: str ='', DM = 0,**kwargs):
        self.target = target
        self.DM = DM
        self.calibrators = []
        super().__init__(**kwargs)

    def from_dict(self,dict: dict) -> None:
        super().from_dict(dict)
        self.target = dict['target']
        if 'DM' in dict:
            self.DM = dict['DM']
        if 'calibrators' in dict:
            self.calibrators = dict['calibrators']
    def to_dict(self):
        dict = super().to_dict()
        dict.update({'target' : self.target})
        dict.update({'DM' : self.DM})
        dict.update({'calibrators' : self.calibrators})
        return(dict)
    
    def __repr__(self):
        return f'ScheduleTrigger({self.time.isot})'

    def plot_kwargs(self) -> dict:
        plotOptions = super().plot_kwargs()
        plotOptions.update({'color':'tab:green'})
        return(plotOptions)

    @property
    def info(self):
        label = 'Trigger\n'
        label += f'Run at: {self.time.isot}\n'
        label += f'Target: {self.target}\n'
        label += f'Status: {self.status}\n'
        label += f'Stations: {self.stations}\n'
        label += f'DM: {self.DM}\n'
        label += f'Calibrators: {self.calibrators}\n'
        label += f'{self.note}\n'
        return(label)

class Station:
    def __init__(self,name: str, servers: list[str] = [], username: str = 'ubuntu', nFile: int = 2,beamformDir: str ='dbaker/daily/',**kwargs) -> None:
        self.name = name
        config = GetStationConfig(self.name)
        self.longName = config['Name']
        self.location = EarthLocation(lon=config['Longitude'],lat=config['Latitude'],height=config['Elevation'])
        self.rotate = config['Rotate_deg']*u.deg
        self.spaceEW = config['Spacing_EW_m']*u.m
        self.spaceNS = config['Spacing_NS_m']*u.m
        self.nRow = config['NRow']
        self.nAnt = config['NAntPerRow']
        self.fLow = config['FreqLim_MHz'][0]*u.MHz
        self.fHigh = config['FreqLim_MHz'][1]*u.MHz
        self.beamPars = {'beam0': 0}
        if 'mainServer' in kwargs:
            self.mainServer=kwargs['mainServer']
        else:
            self.mainServer = servers[0]
        
        if 'delays' in kwargs:
            self.delays = kwargs['delays']
        else:
            self.delays = [0 for row in range(self.nRow-1)]
        if self.nRow<=4:
            self.beamPars.update({f'th{i}': 0 for i in range(self.nRow)})
        else:
            self.beamPars.update({'beam1':0})
        self.servers = servers
        # if (not isinstance(self.mainServer,str)) and len(servers)>0:
        #     self.mainServer = self.servers[-1]
        self.username = username
        self.nFile = nFile
        self.beamformBase = beamformDir


    def get_beam_time(self,time: Time, updates: list[SchedulePoint]) -> dict:
        newBeamPars = {key : {'value' : 0, 'lastUpdate' : Time(0,format='unix')} for key in self.beamPars}
        newDelays = [0 for row in range(self.nRow-1)]
        lastDelayUpdate = Time(0,format='unix')
        for update in updates:
            if self.name in update.stations:
                if update.time<time:
                    for key in update.updates[self.name]:
                        if newBeamPars[key]['lastUpdate']<update.time:
                            newBeamPars[key]['lastUpdate']=update.time
                            newBeamPars[key]['value'] = update.updates[self.name][key]
                    if lastDelayUpdate < update.time:
                        if self.name in update.delays:
                            newDelays = update.delays[self.name]
                            lastDelayUpdate = update.time
        if lastDelayUpdate>Time(0,format='unix'):
            return(newBeamPars,newDelays)
        return(newBeamPars,None)

    def update_beams_to_time(self,time: Time, updates: list[SchedulePoint]) -> None:
        newPars,newDelays = self.get_beam_time(time,updates)
        print(newPars,newDelays)
        for key in self.beamPars:
            if key in newPars:
                self.beamPars[key] = newPars[key]['value']
        if isinstance(newDelays,list):
            self.delays = newDelays

    def run_simulation(self,target,date,padding=5,**kwargs) -> str:
        checkTimes ='/data/kylin/bin/check_times_bf256_target.py'
        if self.nRow <=4:
            sys.argv = ['',f'{self.beamPars["beam0"]}','0',
                                    target,date,'--site',f'{self.name}','--ant','ASIAALPDA','--flim',
                                    f'{int(self.fLow.value)}',f'{int(self.fHigh.value)}','--tlim',f'-{padding}',f'{padding}','--angle',' '.join([str(self.beamPars[x]) for x in self.beamPars if x[:2]=='th'])]
            print(sys.argv)
            with open(checkTimes, "r") as infile:
                exec(''.join(infile.readlines()),{})
            schedFile = f'times_{target}_{date}_{self.longName}_16x{self.nRow}_b0_{self.beamPars["beam0"]:.2f}_0.00_utc_8.txt'
        else:
            sys.argv = ['',f'{self.beamPars["beam0"]}',f'{self.beamPars["beam1"]}',
                        target,date,'--site',f'{self.name}','--ant','ASIAALPDA',
                        '--flim',f'{int(self.fLow.value)}',f'{int(self.fHigh.value)}','--tlim',f'-{padding}',f'{padding}'
                        ]
            with open(checkTimes, "r") as infile:
                exec(''.join(infile.readlines()),{})
            schedFile = f'times_{target}_{date}_{self.longName}_16x{self.nRow}_b0_{self.beamPars["beam0"]:.2f}_{self.beamPars["beam1"]:.2f}_utc_8.txt'
        if any(key in kwargs for key in ['rotation','width','height']):
            fig = plt.gcf()
            ax = fig.get_axes()[0]
            if 'rotation' in kwargs:
                ax.xaxis.set_tick_params(rotation=kwargs['rotation'])
            if 'width' in kwargs:
                fig.set_figwidth(kwargs['width'])
            if 'height' in kwargs:
                fig.set_figwidth(kwargs['height'])
            fig.tight_layout()
            # self.__read_times(f'times_{self.source}_{self.date}_{self.longname}_16x{self.nRow}_b0_{beam0:.2f}_{beam1:.2f}_utc_8.txt')
        return(schedFile)

    def get_server_space(self) -> None:
        self.driveSpace = {}
        for server in self.servers:
            d,s = get_disks(server,username=self.username)
            self.driveSpace.update({server : {'disk': d,'space':s}})

    def transfer_file(self,source: str, destination: str,removeSource: bool = False):
        for server in self.servers:
            command = ["rsync", "-avz"]
            if removeSource and server==self.servers[-1]:
                command+=["--remove-source-files"]
            command+=[source, f'{self.username}@{server}:{destination}']
        
            try:
                # Run the command and capture output
                result = subprocess.run(
                    command, 
                    check=True, 
                    text=True, 
                    capture_output=True
                )
                
            except subprocess.CalledProcessError as e:
                print(f"Rsync failed with exit code {e.returncode}", file=sys.stderr)
                print(f"Error output:\n{e.stderr}", file=sys.stderr)

    def run_remote_command(self,command: str,shell=False,awaitOutputs=True,useMain = False) -> dict:
        outputs = {}
        if useMain:
            iterator = [self.servers[-1]]
        processes = []
        iterator = self.servers
        for server in iterator:
            if 'rsync' in command:
                process = subprocess.Popen(f'ssh -tt {self.username}@{server} {shlex.quote("bash -ic "+shlex.quote(command))}',
                           shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, # Merge stderr to catch errors here too
                        text=True,
                        bufsize=1)
                if not awaitOutputs:
                    processes.append(process)
                    continue
                percent_pattern = re.compile(r'(\d+)%')
                while True:
                        # rsync progress updates often end with '\r' (carriage return), not '\n'
                        # To handle this, we read chunk-by-chunk or character-by-character
                        # Reading lines works well for --info=progress2 as it flushes periodically
                        line = process.stdout.readline()
                        # print(line)
                        if not line and process.poll() is not None:
                            break
                        
                        if line:
                            match = percent_pattern.search(line)
                            if match:
                                percent = match.group(1)
                                # Clean up the output string to show a custom text progress
                                sys.stdout.write(f"\rProgress: [{percent}%] Parsing data...")
                                sys.stdout.flush()
                process.stdout.close()
                process.wait()
                if process.returncode == 0:
                    outputs.update({server: 'rsync complete'})
                else:
                    outputs.update({server: 'rsync failed'})
            else:
                result = subprocess.run(f'ssh -tt {self.username}@{server} {shlex.quote("bash -ic "+shlex.quote(command))}',text=True,capture_output=True,shell=shell)
                print(result.stderr)
                outputs.update({server: result.stdout})
        if awaitOutputs:
            return(outputs)
        else:
            return(processes)

    def submit_event(self,event: ScheduleItem,display=False):
        if self.name in event.stations and event.status == 'pending':
            if isinstance(event,ScheduleBeam):
                if not all([event.id in event.files[x] for x in event.files]):
                    event.strip_files()
                self.transfer_file(event.files[self.name],"/home/ubuntu/shwang/plan/")
                cmd =f'source /data/kylin/bin/mod_activate.sh;conda activate bursttda;'
                cmd += 'python /home/ubuntu/shwang/script/scheduleObs.py beam'
                cmd += f' --stn {self.name}'
                cmd += f' -c /home/ubuntu/shwang/plan/{event.files[self.name]}'
                cmd += f' --start {time_to_string(event.start)} --end {time_to_string(event.end)}'
                cmd += ' --disk_ids'
                for disk in event.drives[self.name]:
                    cmd+=f' {disk}'
                cmd += f' -n {self.nFile}'
                cmd += ' --loop_file /home/ubuntu/shwang/online/loop_obs.txt'
                result = self.run_remote_command(cmd,shell=True)
                if display:
                    print(self.name)
                    print(result)
            if isinstance(event,SchedulePoint):
                if self.name == 'FUS':
                    return
                delays = event.delays[self.name]
                angles = []
                for i in range(self.nRow):
                    if f'th{i}' in event.updates[self.name]:
                        angles.append(event.updates[self.name][f'th{i}'])
                timeStr,timeSleep = time_to_at(event.time)
                cmd = f'/home/ubuntu/shwang/script/set2ndBFM.sh -s {self.spaceNS.to_value(u.m)}'
                for i in range(len(delays)):
                    cmd += f' -d {i+1} {delays[i]}'
                cmd += f' --flim {self.fLow.to_value(u.MHz):.0f}. {self.fHigh.to_value(u.MHz):.0f}.'
                cmd += f" --theta "
                cmd += '"'
                cmd +=' '.join(map(str, angles))
                cmd+= '"'
                cmd = shlex.quote(f'export HOSTNAME=$(hostname); sleep {timeSleep}; {cmd} > /home/ubuntu/Doutput.log 2> /home/ubuntu/Derror.log')
                cmd = f'echo {cmd} | at {timeStr}'
                result = self.run_remote_command(cmd,shell=True)
                if display:
                    print(self.name)
                    print(result)

    # def recover_event(self,event: ScheduleItem, display = False):
    #     if

    def source_angles_at_time(self,time,**kwargs):
        if 'source' in kwargs:
            source = kwargs['source']
        elif 'sourceName' in kwargs:
            source = SkyCoord.from_name(kwargs['sourceName'])
        elif 'ra' in kwargs:
            source = SkyCoord(ra=kwargs['ra'],dec=kwargs['dec'])
        alaz = AltAz(location=self.location,obstime=time)
        sourceALAZ = source.transform_to(alaz)
        thEW,thNS = alaz_to_thetas(sourceALAZ.alt,sourceALAZ.az,self.rotate)
        return(sourceALAZ.alt,sourceALAZ.az,thEW,thNS)


    @property
    def beamformDir(self):
        if self.beamformBase.startswith('/'):
            return(self.beamformBase)
        return(os.path.join(f'/home/{self.username}',self.beamformBase))