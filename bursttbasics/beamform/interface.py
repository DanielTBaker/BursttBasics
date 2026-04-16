import ipywidgets as widgets
import pandas as pd
import beamform as bf
from configparser import ConfigParser
from .bfutils import *
from astropy.time import Time
from astropy.coordinates import SkyCoord
from ..packet_func import *
from .stationclass import Station

class BeamFormer(widgets.Tab):
    def __init__(self,config,**kwargs):
        super().__init__(**kwargs)
        self.station = Station()
        self.make_station_tab()
        self.stationTab.load_ini(config)
        self.singleTab = BeamFormerRun(self.station)
        self.csvTab = CSVTab()
        self.csvTab.run.on_click(self.csv_run)
        self.csvTab.foldRun.on_click(self.csv_fold)
        self.transferTab = TransferWindow(self.station)
        self.children = [self.stationTab,self.singleTab,self.csvTab,self.transferTab]
        self.titles = ['Station','Single','CSV','Transfer']

    def make_station_tab(self):
        self.stationTab = StationProperties()
        stationPars = ["nAnt",
                       "nRow",
                       "xBeam0",
                       "yBeam0",
                       "sepX",
                       "sepY",
                       "skipRows",
                       "nChan",
                       "nOrder",
                       "orderOff",
                       "BW",
                       "f0",
                       "fRef",
                       "byteMeta",
                       "bytePack",
                       "fpp",
                       "fl2int",
                       "beamform",
                       'bwLimit',
                       'targetIP',
                       'stationID',
                       'calFormat',
                       'saveDir'
                      ]
        for s in stationPars:
            getattr(self.stationTab,s).observe(self.change_station_settings,names='value')
        self.stationTab.calFile.observe(self.change_calibrations,names='value')
    def csv_run(self,click):
        self.csvTab.output.clear_output()
        for row in self.csvTab.table.table.children:
            if row.use.button_style=='success':
                eventFiles = find_files(getattr(row,'EventID').value,self.singleTab.dataFolders)
                if self.csvTab.runStyle.value == 'Delete':
                    self.csvTab.output.clear_output()
                    with self.csvTab.output:
                        print('Delete Files:')
                        print(eventFiles)
                    confirm = input()
                    if confirm.lower() in ['y','yes']:
                        print(confirm)
                    continue
                ep_event = getattr(row,'Timestamp').value
                ep_event = ep_event.split('+')[0]
                DM = float(getattr(row,'DM').value)
                if self.csvTab.runStyle.value == 'Center Time':
                    try:
                        getattr(row,'Timestamp').value = self.station.get_center_time(eventFiles,DM,verbose=True,offset=self.csvTab.tOff.value*u.s)
                    except Exception as e:
                        print(e)
                        pass
                    continue
                secWin = self.csvTab.overrideSecWin.value*u.s
                if self.csvTab.runStyle.value == 'Beam Locator':
                    source = getattr(row,'Name_src').value
                    try:
                        test = SkyCoord.from_name(source)
                        with self.csvTab.output:
                            beamID = in_beam_test(test,Time(ep_event,scale='utc'),findClosest=True)
                    except Exception as e:
                        try:
                            test = SkyCoord.from_name(f'PSR {source}')
                            with self.csvTab.output:
                                beamID = in_beam_test(test,Time(ep_event,scale='utc'),findClosest=True)
                        except:
                            with self.csvTab.output:
                                print(f"{getattr(row,'EventID').value}:{self.csvTab.overrideTargetName.value} Not Found")
                            continue
                    getattr(row,'BeamID').value=str(beamID)
                    continue
                if len(self.csvTab.overrideDM.value)>0:
                    try:
                        DM = float(self.csvTab.overrideDM.value)
                    except:
                        pass
                
                if len(self.csvTab.overrideTargetName.value)>0:
                    try:
                        test = SkyCoord.from_name(self.csvTab.overrideTargetName.value)
                        with self.csvTab.output:
                            beamID = in_beam_test(test,Time(ep_event,scale='utc'))
                    except Exception as e:
                        print(e)
                        try:
                            test = SkyCoord.from_name(f'PSR {self.csvTab.overrideTargetName.value}')
                            with self.csvTab.output:
                                beamID = in_beam_test(test,Time(ep_event,scale='utc'))
                        except:
                            with self.csvTab.output:
                                print(f"{getattr(row,'EventID').value}:{self.csvTab.overrideTargetName.value} Not Found")
                            continue
                else:
                    try:
                        beamID = int(getattr(row,'BeamID').value)
                    except:
                        continue
                if beamID<0:
                    with self.csvTab.output:
                        print(f"{getattr(row,'EventID').value}:{self.csvTab.overrideTargetName.value} Not in Beam")
                    continue
                ddSpec,ep0,beamIDX,beamIDY,hasData = self.station.beamform_event(eventFiles,beamID,
                                                        DM,ep_event,
                                                        secWin = secWin)
                if hasData:
                    ddTime = np.arange(ddSpec.shape[1])*self.station.timeFrame # time offset of each frame
                    out_name = f'baseband_beamform_dedisperse_{self.station.stationID}_{getattr(row,"EventID").value}_{beamID}.npz'
                    if self.csvTab.runStyle.value == 'Save':
                        np.savez(os.path.join(self.station.saveDir,out_name), freq=self.station.freq, ep0=ep0,
                                 dd_time=ddTime, dd_spec=ddSpec, ep_event=ep_event,
                                 DM=float(getattr(row,'DM').value), beamid_x=beamIDX, beamid_y=beamIDY)
                    SNR = plot_spectrum(ddSpec,self.station.freq,ddTime.to(u.ms),self.csvTab.output,title=getattr(row,'EventID').value)
                    if self.csvTab.runStyle.value == 'SNR Cut':
                        if SNR<3.5:
                            row.use.button_style='danger'
                else:
                    with self.csvTab.output:
                        print(f'{getattr(row,"EventID").value} has no data')

    def csv_fold(self,click):
        self.csvTab.output.clear_output()
        for row in self.csvTab.table.table.children:
            if row.use.button_style=='success':
                ep0 = getattr(row,'Timestamp').value.split('+')[0]
                kwargs = {}
                if self.csvTab.foldMode == 'radec':
                    kwargs.update({'ra' : self.csvTab.foldRA.value, 'dec': self.csvTab.foldDec.value})
                elif self.csvTab.foldMode == 'beam':
                    kwargs.update({'beamIDF' : self.csvTab.foldBeam.value})
                elif self.csvTab.foldMode == 'source':
                    try:
                        test = SkyCoord.from_name(self.csvTab.foldSource.value)
                        with self.csvTab.output:
                            beamID = in_beam_test(test,Time(ep0,scale='utc'),plot=True)
                        kwargs.update({'beamIDF' : beamID})
                    except:
                        try:
                            test = SkyCoord.from_name(f'PSR {self.csvTab.foldSource.value}')
                            with self.csvTab.output:
                                beamID = in_beam_test(test,Time(ep0,scale='utc'),plot=True)
                            kwargs.update({'beamIDF' : beamID})
                        except:
                            continue
                    if beamID<0:
                        continue
                    
                else:
                    kwargs.update({'sourceName' : f'PSR {self.csvTab.foldSource.value}'})
                eventFiles = find_files(getattr(row,'EventID').value,self.singleTab.dataFolders)
                
                self.profile=self.station.fold_on_target(eventFiles,ep0,
                                                         period = self.csvTab.foldPeriod.value,
                                                         DM = self.csvTab.foldDM.value,
                                                         nBins = self.csvTab.foldBins.value,**kwargs)
                    
                self.profile[self.profile==0]=np.nan
                self.profile/=np.nanmean(self.profile,1)[:,np.newaxis]
                v = np.nanstd(self.profile,1)
                v2 = sigma_clip(v,2)
                self.profile[v2.mask]=np.nan
                y = self.station.freq.value
                dy=y[1]-y[0]
                ext = [-.5,.5,y[0]-dy/2,y[-1]-dy/2]
                grid=plt.GridSpec(nrows=2,ncols=1,height_ratios=[4,1],hspace=0)
                with self.csvTab.output:
                    phase = np.linspace(-.5,.5,2*self.profile.shape[1]+1)[1:-1:2]
                    prof1D = np.nanmean(self.profile,0)
                    plt.figure()
                    plt.suptitle(getattr(row,'EventID').value)
                    plt.subplot(grid[0])
                    plt.imshow(self.profile,origin='lower',aspect='auto',
                               extent = ext,
                               vmin=1,vmax=1+2*np.nanstd(self.profile))
                    plt.ylabel('Freq (MHz)')
                    plt.xticks([])
                    ax=plt.subplot(grid[1])
                    plt.plot(phase,prof1D)
                    peakPhase = phase[np.argmax(prof1D)]
                    plt.axvline(phase[np.argmax(prof1D)])
                    plt.axvline(peakPhase,c='k',ls='--')
                    height = ax.get_ybound()[1] - ax.get_ybound()[0]
                    width = ax.get_xbound()[1] - ax.get_xbound()[0]
                    if peakPhase<0:
                        plt.text(peakPhase,ax.get_ybound()[1]-height*.2,f'Off = {peakPhase:.3f}')
                    else:
                        plt.text(peakPhase-width*.15,ax.get_ybound()[1]-height*.2,f'Off = {peakPhase*self.csvTab.foldPeriod.value:.3f}s')
                    plt.xlim(-.5,.5)
                    plt.xlabel('Pulse Phase')
                    plt.tight_layout()
                    plt.show()
                row.use.button_style='danger'
                
    def change_station_settings(self, change):
        name = change['owner'].description
        new = change['new']
        if name in ["BW",
                    "f0",
                    "fRef"]:
            new*=u.MHz
        if "sep" in name:
            new*=u.m
        if name == 'beamform':
            new = getattr(bf,new)
        setattr(self.station,name,new)

    def change_calibrations(self,change):
        self.station.set_cal_options(change['new'])

class TableRow(widgets.HBox):
    def __init__(self,data,keys,widths):
        super().__init__(layout=widgets.Layout(border='2px'))
        for keyID,key in enumerate(keys):
            newItem = widgets.Label(str(data[key]),layout=widgets.Layout(border='1px dashed',
                                                                              width=f'{widths[keyID]}px',
                                                                              min_height = '25px',
                                                                              margin='0px 0px 0px 0px',
                                                                              justify_content="center"))
            setattr(self,str(key),newItem)
        self.delete = widgets.Button(icon = "solid trash",layout=widgets.Layout(width=f'50px',
                                                                              min_height = '25px',
                                                                              margin='0px 0px 0px 0px',
                                                                              justify_content="center"))
        self.delete.parent = self
        self.use = widgets.Button(button_style='success',layout=widgets.Layout(width=f'50px',
                                                                              min_height = '25px',
                                                                              margin='0px 0px 0px 0px',
                                                                              justify_content="center"))
        self.use.on_click(self.toggle_button)
        
        self.children = [getattr(self,str(key)) for key in keys] + [self.delete,self.use]

    def toggle_button(self,button):
        if button.button_style == 'success':
            button.button_style = 'danger'
        else:
            button.button_style = 'success'
            

class WidgetTable(widgets.VBox):
    def __init__(self,headers,widths,nRows = 10,**kwargs):
        super().__init__(**kwargs)
        self.headersBox = widgets.HBox()
        self.widths = widths
        self.headers = headers
        for headerID,header in enumerate(self.headers):
            self.headersBox.children = list(self.headersBox.children) + [widgets.Label(header,
                                                       layout=widgets.Layout(width=f'{self.widths[headerID]}px',
                                                                             margin='0px 0px 0px 0px',
                                                                             justify_content="center"))]
        self.headersBox.children=list(self.headersBox.children) + [widgets.Label("Delete",layout=widgets.Layout(width=f'50px',
                                                                             margin='0px 0px 0px 0px',
                                                                             justify_content="center"))]
        self.headersBox.children=list(self.headersBox.children) + [widgets.Label("Use",layout=widgets.Layout(width=f'50px',
                                                                             margin='0px 0px 0px 0px',
                                                                             justify_content="center"))]
        self.table = widgets.VBox(layout = widgets.Layout(max_height = f'{25*nRows}px',overflow_y='scroll',display='block'))

        self.newEntry = widgets.HBox()
        for headerID,header in enumerate(self.headers):
            self.newEntry.children = list(self.newEntry.children) + [widgets.Text(layout=widgets.Layout(width=f'{self.widths[headerID]}px',
                                                                             margin='0px 0px 0px 0px',
                                                                             justify_content="center"))]
        self.addEntry = widgets.Button(description='Add',layout=widgets.Layout(width=f'100px',
                                                                             margin='0px 0px 0px 0px',
                                                                             justify_content="center"),icon='plus')
        
        self.newEntry.children = list(self.newEntry.children) + [self.addEntry]
        self.addEntry.on_click(self.add_custom)
        
        self.fileBox = widgets.HBox()
        self.fileName = widgets.Text(description='File')
        self.load = widgets.Button(description='Load',icon='upload')
        self.save = widgets.Button(description='Save',icon='download')
        self.append =  widgets.Button(description='Append',icon='plus')
        self.append.on_click(self.append_click)
        self.save.on_click(self.save_click)
        self.load.on_click(self.load_click)
        self.fileBox.children = [self.fileName,self.save,self.load,self.append]
        
        self.children = [self.headersBox,self.table,self.newEntry,self.fileBox]
        
    def add_custom(self,click):
        data = {}
        for hid,header in enumerate(self.headers):
            data.update({header : self.newEntry.children[hid].value})
        self.add_row(data)
        
    def add_row(self,data):
        newRow = TableRow(data,self.headers,self.widths)
        newRow.delete.on_click(self.delete_click)
        self.table.children = list(self.table.children) + [newRow]

    def remove_row(self,row):
        if isinstance(row,TableRow):
            try:
                children = list(self.table.children)
                children.remove(row)
                self.table.children = children
            except:
                print('Row not in table')
        elif isinstance(row,int):
            if row<len(self.table.children):
                children = list(self.table.children)
                del children[row]
                self.table.children = children

    def delete_click(self,click):
        if hasattr(click,'parent'):
            self.remove_row(click.parent)

    def rows_from_csv(self,fileName):
        try:
            df = pd.read_csv(fileName)
        except:
            print('Cannot Load File')
            return
        for rowID,row in df.iterrows():
            try:
                self.add_row(row)
            except:
                pass

    def load_click(self,click):
        self.table.children = []
        self.rows_from_csv(self.fileName.value)

    def append_click(self,click):
        self.rows_from_csv(self.fileName.value)

    def save_click(self,click):
        data={}
        for header in self.headers:
            data.update({header : [getattr(row,header).value for row in self.table.children]})
        df= pd.DataFrame(data)
        try:
            df.to_csv(self.fileName.value)
        except:
            print('cannot save to filename')
            
    @classmethod
    def from_csv(cls, fileName,widths=None):
        df = pd.read_csv(fileName)
        headers = [x for x in df.columns]
        if widths == None:
            widths = [1000//len(headers) for i in range(len(headers))]
        table = cls(headers,widths)
        for rowID,row in df.iterrows():
            table.add_row(row)
        return table
    
class StationProperties(widgets.GridspecLayout):
    def __init__(self,n_columns=3,n_rows = 16,**kwargs):
        super().__init__(n_columns=n_columns,n_rows = n_rows,**kwargs)
        currentRow = 0
        self[currentRow,:] = widgets.Label('General' )
        currentRow+=1
        self.stationID = widgets.Text(description='stationID',continuous_update=False)
        self.targetIP = widgets.Text(description='targetIP',continuous_update=False)
        self.bwLimit = widgets.IntText(description='bwLimit',min=1,continuous_update=False,layout=widgets.Layout(width='20ch'))
        self.Mlabel = widgets.Label(value='M')
        self.bwBox = widgets.HBox()
        self.bwBox.children = [self.bwLimit,self.Mlabel]
        self.saveDir = widgets.Text(description='saveDir',continuous_update=False)
        self[currentRow,0]=self.stationID
        self[currentRow,1]=self.targetIP
        self[currentRow,2]=self.bwBox
        currentRow+=1
        self[currentRow,:]=self.saveDir
        currentRow+=1
        self[currentRow,:] = widgets.Label('Geometry' )
        currentRow+=1
        self.nAnt = widgets.IntSlider(value=16,min=1,max=64,description='nAnt',readout=True)
        self.nRow = widgets.IntSlider(value=16,min=1,max=64,description='nRow',readout=True)
        self.xBeam0 = widgets.FloatText(value=-7.5,description='xBeam0',continuous_update=False)
        self.yBeam0 = widgets.FloatText(value=-7.5,description='yBeam0',continuous_update=False)
        self.sepX = widgets.FloatText(value=1.0,description='sepX',continuous_update=False)
        self.sepY = widgets.FloatText(value=0.5,description='sepY',continuous_update=False)
        self.skipRows = widgets.IntsInput(description = 'skipRows',value=[],min=0,max=15,allow_duplicates=False)
        self.nRow.observe(self.update_skip_row_max,names='value')
        self[currentRow,0]=self.nAnt
        self[currentRow,1]=self.nRow
        currentRow+=1
        self[currentRow,0]=self.xBeam0
        self[currentRow,1]=self.yBeam0
        currentRow+=1
        self[currentRow,0]=self.sepX
        self[currentRow,1]=self.sepY
        currentRow+=1
        self[currentRow,:]= widgets.Label('Skip Rows:')
        currentRow+=1
        self[currentRow,:]=self.skipRows
        currentRow+=1
        self[currentRow,:] =  widgets.Label('Spectral' )
        currentRow+=1
        self.nChan = widgets.IntSlider(value=1024,min=128,max=2048,step=128,description='nChan',readout=True)
        self.nOrder = widgets.IntSlider(value=8,min=1,max=16,description='nOrder',readout=True)
        self.orderOff = widgets.IntText(value=0,description='orderOff')
        self.BW = widgets.FloatText(value=400,description='BW',continuous_update=False)
        self.f0 = widgets.FloatText(value=400,description='f0',continuous_update=False)
        self.fRef = widgets.FloatText(value=400,description='fRef',continuous_update=False)
        self[currentRow,0]=self.nChan
        self[currentRow,1]=self.nOrder
        self[currentRow,2]=self.orderOff
        currentRow+=1
        self[currentRow,0]=self.f0
        self[currentRow,1]=self.BW
        self[currentRow,2]=self.fRef
        currentRow+=1
        self[currentRow,:]=widgets.Label('Format')
        currentRow+=1
        self.byteMeta = widgets.IntSlider(value=64,min=1,max=128,description='byteMeta',readout=True)
        self.bytePack = widgets.IntSlider(value=8256,min=1,max=1024*16+64,description='bytePack',readout=True)
        self.fpp = widgets.IntSlider(value=4,min=1,max=16,description='fpp',readout=True)
        self.fl2int = widgets.IntSlider(value=256,min=1,max=1024,description='fl2int',readout=True)
        self.beamform = widgets.Dropdown(options=[x for x in dir(bf) if 'beamform' in x],value='beamform256',description='beamform')
        self.calFile = widgets.Text(value = "./calibration", description = "calFolder",
                                    layout=widgets.Layout(display='flex',width='80%'),continuous_update=False)
        self.calFormat = widgets.Dropdown(options=['.txt','.npz'],description = 'calFormat')
        self[currentRow,0] = self.byteMeta
        self[currentRow,1] = self.bytePack
        currentRow+=1
        self[currentRow,0] = self.fpp
        self[currentRow,1] = self.fl2int
        currentRow+=1
        self[currentRow,0] = self.beamform
        self[currentRow,1] = self.calFile
        self[currentRow,2] = self.calFormat

    def update_skip_row_max(self,change):
        self.skipRows.max = change['new']-1
        self.skipRows.value = [x for x in self.skipRows.value if x<=change['new']-1]
        
    def load_ini(self,fileName):
        conf = ConfigParser()
        conf.read(fileName)
        self.nAnt.value = conf['Geometry'].getint("nAnt",16)
        self.nRow.value = conf['Geometry'].getint("nRow",16)
        self.xBeam0.value = conf['Geometry'].getfloat("xBeam0",-7.5)
        self.yBeam0.value = conf['Geometry'].getfloat("yBeam0",-7.5)
        self.sepX.value = conf['Geometry'].getfloat("sepX",1.0)
        self.sepY.value = conf['Geometry'].getfloat("sepY",0.5)

        self.nChan.value = conf['Spectral'].getint("nChan",1024)
        self.nOrder.value = conf['Spectral'].getint("nOrder",8)
        self.orderOff.value = conf['Spectral'].getint('orderOff',0)
        self.f0.value = conf['Spectral'].getfloat("f0",400)
        self.BW.value = conf['Spectral'].getfloat("BW",400)
        self.fRef.value = conf['Spectral'].getfloat("fRef",400)

        self.byteMeta.value = conf['Format'].getint("byteMeta",64)
        self.bytePack.value = conf['Format'].getint("bytePack",8256)
        self.fpp.value = conf['Format'].getint("fpp",4)
        self.fl2int.value = conf['Format'].getint("fl2int",256)
        self.beamform.value = conf['Format']['beamform']
        self.calFile.value = conf['Format']['calFolder']
        self.calFormat.value = conf['Format'].get('calFormat','.txt')
        self.stationID.value = conf['General']['stationID']
        self.targetIP.value = conf['General']['transferIP']
        self.bwLimit.value = conf['General'].getint('bwLimit',1)
        self.saveDir.value = conf['General'].get('saveDir','./')


class BeamFormerRun(widgets.VBox):
    def __init__(self,station,**kwargs):
        super().__init__(**kwargs)
        self.dataFolders = sorted(glob("/burstt[0-9]/disk*/data*")+glob("/burstt1[0-9]/disk*/data*"))
        self.station=station
        self.eventBox = widgets.HBox()
        self.triggerRefresh = widgets.Button(icon='refresh',layout=widgets.Layout(width='5ch'))
        self.triggerRefresh.on_click(self.refresh_triggers)
        events = np.sort(np.unique(np.array([x.split('/')[-1][6:-12] for x in glob("/burstt[0-9]/disk*/data*/ring*Z.trig*")+glob("/burstt1[0-9]/disk*/data*/ring*Z.trig*")])))
        self.eventID = widgets.Dropdown(options=events,value=events[0],descrition='eventID')
        self.ep0 = widgets.Text(value = time_from_event(self.eventID.value),description='ep0',continuous_update=False)
        self.eventBox.children = [self.triggerRefresh,self.eventID,self.ep0]
        
        self.eventID.observe(self.update_time,names='value')
        self.eventFiles = find_files(self.eventID.value,self.dataFolders)

        self.accordion = widgets.Accordion()
        
        self.baseTab = widgets.VBox()
        self.baseBox = widgets.GridspecLayout(n_columns=3,n_rows =6)
        self.baseOut = widgets.Output()
        self.baseDM = widgets.FloatText(value=0,description='DM',layout=widgets.Layout(width='25ch'))
        self.baseBeam = widgets.IntText(value=0,min=0,max=255,description='beamID',layout=widgets.Layout(width='25ch'))
        self.baseSecWin = widgets.FloatText(value=0.03,description='Window',layout=widgets.Layout(width='25ch'))
        self.baseTimeOff = widgets.FloatText(value = 0, description = 'ep0 Offset',layout=widgets.Layout(width='25ch'))
        self.baseRun = widgets.Button(description = 'RUN',layout=widgets.Layout(width='25ch'))
        self.baseRebin = widgets.IntText(value=400,min=1,description='Time Rebin',layout=widgets.Layout(width='25ch'))
        self.baseSourceType = widgets.Dropdown(description='Source Type:',options=['Beam','Target'],value='Beam')
        self.baseSourceType.observe(self.update_base_source,names='value')
        self.baseBeamTest = widgets.Button(description='Test Beam',layout=widgets.Layout(width='25ch'))
        self.baseBeamTest.on_click(self.test_beam)
        self.baseTargetName = widgets.Combobox(description='Target:',options=['B0531+21','B0329+54','3C147','3C48','3C273'])
        self.baseTargetName.observe(self.base_target_change,names='value')
        self.firstOnly = widgets.Checkbox(description = 'First Beamform')
        self.skipOrders = widgets.IntsInput()
        self.baseBox[0,0]=self.baseDM
        self.baseBox[0,1]=self.baseSecWin
        self.baseBox[0,2]=self.baseTimeOff
        self.baseBox[1,:]=self.baseSourceType
        self.baseBox[2,1]=self.baseBeam
        self.baseBox[3,:]=widgets.Label('Skip Orders:')
        self.baseBox[4,:]=self.skipOrders
        self.baseBox[5,0]=self.firstOnly
        self.baseBox[5,1]=self.baseRun
        self.baseTab.children = [self.baseBox,self.baseOut]
        self.baseRun.on_click(self.run_base)

        self.foldTab = widgets.VBox()
        self.foldBox = widgets.HBox()
        self.foldOut = widgets.Output()
        self.foldDM = widgets.FloatText(value=0,description='DM',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldPeriod = widgets.FloatText(value=1,description='Period',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldBins = widgets.IntText(value=256,min=1,description='nBins',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldRun = widgets.Button(description = 'Run',layout=widgets.Layout(width='25ch'),icon='play')
        self.foldRun.on_click(self.run_fold)
        self.foldBox.children = [self.foldDM,self.foldPeriod,self.foldBins,self.foldRun]
        self.foldSourceBox = widgets.HBox()
        self.foldRA = widgets.FloatText(value=0,description= 'RA',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldDec = widgets.FloatText(value=0,description= 'DEC',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldMode = "radec"
        self.foldBeam = widgets.IntText(value=0,min=0,max=255,description='beamID',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldSource = widgets.Text(description='Source',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldSourceBox.children = [self.foldRA,self.foldDec,self.foldBeam,self.foldSource]
        for child in self.foldSourceBox.children:
             child.observe(self.fold_target_change,names='value')

        self.foldTab.children = [self.foldBox,self.foldSourceBox,self.foldOut]

        # self.calibratorTab = widgets.VBox()
        # self.calibratorBox = widgets.HBox()
        # self.calibratorOut = widgets.Output()
        # self.calibratorOptions = widgets.Dropdown(options=['Max','Fixed'],value='Max',description='Cut Mode')
        # self.calibratorBeam = widgets.IntText(value=0,min=0,
        #                                       max=255,description='beamID',
        #                                       layout=widgets.Layout(width='20ch'))
        # self.calibratorEp0 = widgets.Text(value = "",
        #                                   description='ep0',continuous_update=False)
        # self.calibratorSecWin = widgets.FloatText(value=0,min=0,description='Window',layout=widgets.Layout(width='25ch'))
        # self.calibratorBox.children=[self.calibratorBeam,self.calibratorOptions]
        # self.calibratorRun = widgets.Button(description = 'Run',icon='play')
        # self.calibratorRun.on_click(self.run_calibrator)
        # self.calibratorTab.children=[self.calibratorBox,self.calibratorRun,self.calibratorOut]
        
        # self.calibratorOptions.observe(self.changeCalMode,names='value')
        
        
        
        self.accordion.children = [self.baseTab,self.foldTab]#,self.calibratorTab]
        self.accordion.titles = ['Single','Fold']#,'Calibrator']
        self.children = [self.eventBox,self.accordion]

    def refresh_triggers(self,click):
        events = np.sort(np.unique(np.array([x.split('/')[-1][6:-12] for x in glob("/burstt[0-9]/disk*/data*/ring*Z.trig*")+glob("/burstt1[0-9]/disk*/data*/ring*Z.trig*")])))
        self.eventID.options = events
        
    def run_calibrator(self,click):
        epPulse = None
        secWin = None
        if self.calibratorSecWin.value>0:
            secWin = self.calibratorSecWin.value
        if len(self.calibratorEp0.value)>0:
            epPulse = self.calibratorEp0.value
        self.calibratorOut.clear_output()
        with self.calibratorOut:
            rb=400
            ddSpec,ep0,beamIDX,beamIDY = self.station.calibrator_beamform(self.eventFiles,
                                             self.calibratorBeam.value,
                                             find_cal=True,
                                             epPulse=epPulse,
                                             secWin=secWin,
                                             pad=.01*u.s)
            ddTime = np.arange(ddSpec.shape[1])*self.station.timeFrame # time offset of each frame
            out_name = f'baseband_beamform_calibrator_{self.eventID.value}_{self.calibratorBeam.value}.npz'
            np.savez(os.path.join(self.station.saveDir,out_name), freq=self.station.freq, ep0=ep0,
                     dd_time=ddTime, dd_spec=ddSpec, ep_event=self.ep0.value,
                     DM=0, beamid_x=beamIDX, beamid_y=beamIDY)
            plot_spectrum(ddSpec,self.station.freq,ddTime.to(u.ms),title=self.eventID.value)
            # im = np.abs(ddSpec)**2
            # im = np.reshape(im[:,:rb*(im.shape[1]//rb)],(im.shape[0],-1,rb)).mean(2)
            # im/=np.nanmean(im,1)[:,np.newaxis]
            # v = np.nanvar(im,1)
            # v2 = sigma_clip(v,2)
            # im[v2.mask]=np.nan
            # x = np.linspace(.5,im.shape[1]+.5,im.shape[1],endpoint=False)*self.station.timeFrame.to_value(u.ms)*rb
            # y = self.station.freq.value
            # dx = x[1]-x[0]
            # dy=y[1]-y[0]
            # ext = [x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2]
            # plt.figure()
            # plt.imshow(im,origin='lower',aspect='auto',
            #            extent=ext,
            #            vmin=1,vmax=1+2*np.nanstd(im))
            # plt.ylabel('Freq (MHz)')
            # plt.xlabel('Time (ms)')
            # plt.show()
        
    def changeCalMode(self,change):
        if change['new'] == 'Max':
            self.calibratorBox.children=[self.calibratorBeam,self.calibratorOptions]
        else:
            self.calibratorBox.children=[self.calibratorBeam,self.calibratorOptions,self.calibratorEp0,self.calibratorSecWin]

    def base_target_change(self,change):
        if change['new'] == 'B0329+54':
            self.baseDM.value = 26.7641
        if change['new'] == 'B0531+21':
            self.baseDM.value = 56.7712
            
    def fold_target_change(self,change):
        if (change['owner'] == self.foldRA) or (change['owner'] == self.foldDec):
            self.foldMode = "radec"
        if change['owner'] == self.foldBeam:
            self.foldMode = 'beam'
        if change['owner'] == self.foldSource:
            
            self.foldOut.clear_output()
            with self.foldOut:
                try:
                    test = SkyCoord.from_name(self.foldSource.value)
                    self.foldMode = 'source'
                    determine_beam(test,Time(self.ep0.value,scale='utc'))
                except:
                    try:
                        test = SkyCoord.from_name(f'PSR {self.foldSource.value}')
                        self.foldMode = 'sourcePSR'
                        determine_beam(test,Time(self.ep0.value,scale='utc'))
                    except:
                        print('Source Not Found')
    def run_fold(self,click):
        self.foldOut.clear_output()
        with self.foldOut:
            kwargs = {}
            if self.foldMode == 'radec':
                kwargs.update({'ra' : self.foldRA.value, 'dec': self.foldDec.value})
            elif self.foldMode == 'beam':
                kwargs.update({'beamIDF' : self.foldBeam.value})
            elif self.foldMode == 'source':
                kwargs.update({'sourceName' : self.foldSource.value})
            else:
                kwargs.update({'sourceName' : f'PSR {self.foldSource.value}'})
            self.profile = self.station.fold_on_target(self.eventFiles,
                                                  self.ep0.value,
                                                  period = self.foldPeriod.value*u.s,
                                                  DM = self.foldDM.value,
                                                  nBins = self.foldBins.value,**kwargs)
            self.profile[self.profile==0]=np.nan
            self.profile/=np.nanmean(self.profile,1)[:,np.newaxis]
            v = np.nanvar(self.profile,1)
            v2 = sigma_clip(v,2)
            self.profile[v2.mask]=np.nan
            y = self.station.freq.value
            dy=y[1]-y[0]
            self.foldOut.clear_output()
            ext = [-.5,.5,y[0]-dy/2,y[-1]-dy/2]
            grid=plt.GridSpec(nrows=2,ncols=1,height_ratios=[4,1],hspace=0)
            plt.figure()
            plt.subplot(grid[0])
            plt.imshow(self.profile,origin='lower',aspect='auto',
                       extent = ext,
                       vmin=1,vmax=1+2*np.nanstd(self.profile))
            plt.ylabel('Freq (MHz)')
            plt.xticks([])
            plt.subplot(grid[1])
            SNR = np.nanmean(self.profile,0)
            SNR -= np.nanmedian(SNR)
            SNR /= np.nanstd(SNR)
            plt.plot(np.linspace(-.5,.5,2*self.profile.shape[1]+1)[1:-1:2],SNR)
            plt.xlim(-.5,.5)
            plt.xlabel('Pulse Phase')
            plt.ylabel('SNR')
            plt.tight_layout()
            plt.show()
            
    def test_beam(self,click):
        self.baseOut.clear_output()
        with self.baseOut:
            try:
                test = SkyCoord.from_name(self.baseTargetName.value)
                targetBeam = determine_beam(test,Time(self.ep0.value,scale='utc'))
            except:
                try:
                    test = SkyCoord.from_name(f'PSR {self.baseTargetName.value}')
                    targetBeam = determine_beam(test,Time(self.ep0.value,scale='utc'))
                except:
                    print('Source Not Found')

    def run_base(self,click):
        rb=self.baseRebin.value
        self.baseOut.clear_output()
        with self.baseOut:
            if self.baseSourceType.value == 'Beam':
                targetBeam = self.baseBeam.value
            if self.baseSourceType.value == 'Target':
                try:
                    test = SkyCoord.from_name(self.baseTargetName.value)
                    targetBeam = determine_beam(test,Time(self.ep0.value,scale='utc'))
                except:
                    try:
                        test = SkyCoord.from_name(f'PSR {self.baseTargetName.value}')
                        targetBeam = determine_beam(test,Time(self.ep0.value,scale='utc'))
                    except:
                        print('Source Not Found')
                        return
            if self.firstOnly.value:
                self.baseOut.clear_output()
                beamIDX,sinThetaIDY,_ =  self.station.beamIDs_from_FUS(targetBeam)
                for beamIDY in range(self.station.nRow):
                    ddSpec,ep0,beamIDX,beamIDY,hasData = self.station.beamform_event_first(self.eventFiles,beamIDX,beamIDY,
                                                            self.baseDM.value,self.ep0.value,
                                                            secWin = self.baseSecWin.value*u.s,
                                                                timeOffset = self.baseTimeOff.value*u.s,skipOrder = self.skipOrders.value)
                    if hasData:
                        ddTime = np.arange(ddSpec.shape[1])*self.station.timeFrame # time offset of each frame
                        out_name = f'baseband_first_dedisperse_{self.station.stationID}_{self.eventID.value}_{beamIDX}_{beamIDY}.npz'
                        np.savez(os.path.join(self.station.saveDir,out_name), freq=self.station.freq, ep0=ep0,
                                 dd_time=ddTime, dd_spec=ddSpec, ep_event=self.ep0.value,
                                 DM=self.baseDM.value, beamid_x=beamIDX, beamid_y=beamIDY)
                        
                        plot_spectrum(ddSpec,self.station.freq,ddTime.to(u.ms),self.baseOut,title=self.eventID.value)
                    else:
                        print(f'{self.eventID.value} has no Data')
                        break
                    
            else:
                ddSpec,ep0,beamIDX,beamIDY,hasData = self.station.beamform_event(self.eventFiles,targetBeam,
                                                            self.baseDM.value,self.ep0.value,
                                                            secWin = self.baseSecWin.value*u.s,
                                                                timeOffset = self.baseTimeOff.value*u.s,skipOrder = self.skipOrders.value)
                self.baseOut.clear_output()
                if hasData:
                    ddTime = np.arange(ddSpec.shape[1])*self.station.timeFrame # time offset of each frame
                    out_name = f'baseband_beamform_dedisperse_{self.station.stationID}_{self.eventID.value}_{targetBeam}.npz'
                    np.savez(os.path.join(self.station.saveDir,out_name), freq=self.station.freq, ep0=ep0,
                             dd_time=ddTime, dd_spec=ddSpec, ep_event=self.ep0.value,
                             DM=self.baseDM.value, beamid_x=beamIDX, beamid_y=beamIDY)
                    
                    plot_spectrum(ddSpec,self.station.freq,ddTime.to(u.ms),self.baseOut,title=self.eventID.value)
                else:
                    print(f'{self.eventID.value} has no Data')
            
    def update_time(self,change):
        self.ep0.value = time_from_event(change['new'])
        self.eventFiles = find_files(change['new'],self.dataFolders)
    def update_base_source(self,change):
        if change['new'] == 'Beam':
            self.baseBox[2,1]=self.baseBeam
        if change['new'] == 'Target':
            self.baseBox[2,1]=self.baseTargetName
            self.baseBox[2,2]=self.baseBeamTest
            

class CSVTab(widgets.VBox):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.table = WidgetTable(['EventID','Timestamp','Name_src','DM','BeamID'],widths = [200,250,150,150,150])
        self.overrideBox = widgets.HBox()
        self.overrideDM = widgets.Text(value=None,description='DM',layout=widgets.Layout(width='25ch'))
        self.overrideSecWin = widgets.FloatText(value=0.06,description='Window',layout=widgets.Layout(width='25ch'))
        self.overrideTargetName = widgets.Combobox(description='Target:',options=['B0531+21','B0329+54','3C147','3C48','3C273'])
        self.overrideBox.children = [self.overrideDM,self.overrideSecWin,self.overrideTargetName]
        self.runStyle = widgets.RadioButtons(description='Run Style:',options=['Save','SNR Cut','Beam Locator','Center Time','Delete'])
        self.tOff = widgets.FloatText(description='Center Offset',value=0)
        self.run = widgets.Button(description='Run',layout=widgets.Layout(flex='1 1 auto', width='auto'),icon='play')
        
        self.foldTab = widgets.VBox()
        self.foldBox = widgets.HBox()
        self.foldDM = widgets.FloatText(value=0,description='DM',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldPeriod = widgets.FloatText(value=1,description='Period',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldBins = widgets.IntText(value=256,min=1,description='nBins',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldRun = widgets.Button(description = 'Run',layout=widgets.Layout(width='25ch'),icon='play')
        # self.foldRun.on_click(self.run_fold)
        self.foldBox.children = [self.foldDM,self.foldPeriod,self.foldBins,self.foldRun]
        self.foldSourceBox = widgets.HBox()
        self.foldRA = widgets.FloatText(value=0,description= 'RA',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldDec = widgets.FloatText(value=0,description= 'DEC',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldMode = "radec"
        self.foldBeam = widgets.IntText(value=0,min=0,max=255,description='beamID',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldSource = widgets.Text(description='Source',layout=widgets.Layout(width='25ch'),continuous_update=False)
        self.foldTab.children = [self.foldBox,self.foldSourceBox]
        self.foldSourceBox.children = [self.foldRA,self.foldDec,self.foldBeam,self.foldSource]
        for child in self.foldSourceBox.children:
             child.observe(self.fold_target_change,names='value')
        
        self.output = widgets.Output()
        

        self.checkBox = widgets.GridspecLayout(n_columns=3,n_rows=1)
        self.minOrder = widgets.IntText(description='Minimum Orders',value=0)
        self.clipSource = widgets.Combobox(description='Target:',options=['B0531+21','B0329+54','3C147','3C48','3C273'])
        self.clipButton = widgets.Button(description='Prune',icon='scissors')
        self.clipButton.on_click(self.prune)
        self.checkBox[0,0]=self.minOrder
        self.checkBox[0,1]=self.clipSource
        self.checkBox[0,2]=self.clipButton
        self.children = [self.table,self.overrideBox,self.runStyle,self.tOff,self.run,self.foldTab,self.checkBox,self.output]
        
    def prune(self,click):
        dataFolders = sorted(glob("/burstt[0-9]/disk*/data*")+glob("/burstt1[0-9]/disk*/data*"))
        clipRows = []
        for rowID,row in enumerate(self.table.table.children):
            eventID = getattr(row,'EventID').value
            epEvent = getattr(row,'Timestamp').value
            epEvent = epEvent.split('+')[0]
            ep0=Time(epEvent)
            files = find_files(eventID,dataFolders)
            useable = []
            for i in range(len(files)):
                fbin = files[i]
                try:
                    sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
                except (OSError, FileNotFoundError) as e:
                    continue
                if sz == 0:
                    continue
                try:
                    with open(fbin, 'rb') as fh:
                        mt = fh.read(64)
                        hd = fh.read(64)
                        tmp = decHeader2(hd)
                        od = tmp[4]
                except:
                    continue
                useable.append(int(od))
            useable = np.unique(np.array(useable)).shape[0]
            if useable<self.minOrder.value:
                clipRows.append(rowID)
                continue
            if len(self.clipSource.value)>0:
                try:
                    test = SkyCoord.from_name(self.clipSource.value)
                    beamID = in_beam_test(test,ep0)
                except:
                    try:
                        test = SkyCoord.from_name(f'PSR {self.clipSource.value}')
                        beamID = in_beam_test(test,ep0)
                    except:
                        continue
                if beamID<0:
                    clipRows.append(rowID)
        clipRows = np.sort(np.array(clipRows))[::-1].astype(int)
        with self.output:
            print(f'Clip Rows: {clipRows}')
        for rowID in clipRows:
            self.table.remove_row(int(rowID))
            
    def fold_target_change(self,change):
        if (change['owner'] == self.foldRA) or (change['owner'] == self.foldDec):
            self.foldMode = "radec"
        if change['owner'] == self.foldBeam:
            self.foldMode = 'beam'
        if change['owner'] == self.foldSource:
            
            self.output.clear_output()
            with self.output:
                try:
                    test = SkyCoord.from_name(self.foldSource.value)
                    self.foldMode = 'source'
                except:
                    try:
                        test = SkyCoord.from_name(f'PSR {self.foldSource.value}')
                        self.foldMode = 'source'
                    except:
                        print('Source Not Found')

class TransferWindow (widgets.VBox):
    def __init__(self,station,**kwargs):
        self.station = station
        super().__init__(**kwargs)
        self.refreshButton = widgets.Button(description='Refresh',icon='refresh',layout=widgets.Layout(width='25ch'))
        self.deleteButton = widgets.Button(description='Delete',icon='trash',layout=widgets.Layout(width='25ch'))
        self.sendButton = widgets.Button(description='Transfer',icon='paper-plane',layout=widgets.Layout(width='25ch'))
        self.buttonBox = widgets.HBox()
        self.buttonBox.children=[self.refreshButton,self.deleteButton,self.sendButton]
        
        self.files = widgets.SelectMultiple(
            options=[],
            value=[],
            description='Files:',layout=widgets.Layout(width='75ch')
        )
        self.update_files(None)
        self.refreshButton.on_click(self.update_files)
        self.deleteButton.on_click(self.delete_files)
        self.sendButton.on_click(self.transfer_files)
        self.output = widgets.Output()
        self.progress = widgets.Output()
        self.targetDir = widgets.Combobox(description='Target Dir:',values=['/burstt_home/shared'],layout=widgets.Layout(width='75ch'))
        self.children = [self.buttonBox,self.files,self.targetDir,self.output,self.progress]
        
        
    def update_files(self,click):
        options = np.sort(np.array([x for x in os.listdir(self.station.saveDir) if ".npz" in x]))
        self.files.options=list(options)

    def delete_files(self,click):
        values = self.files.value
        options = list(self.files.options)
        for file in values:
            os.system(f'rm {file}')
            options.remove(file)
        self.files.options = options
        self.files.value = []
            
    def transfer_files(self,click):
        self.output.clear_output()
        self.progress.clear_output()
        fileList = list(self.files.value)
        fileList = [os.path.join(self.station.saveDir,f) for f in fileList]
        if len(fileList)==0:
            with self.output:
                print('No Files to Transfer')
            return
        files = ' '.join(fileList)
        bwLimit = f'{self.station.bwLimit}M'
        targetIP = self.station.targetIP
        targetDir = self.targetDir.value
        destination = f'ubuntu@{targetIP}:{targetDir}'
        command = ["rsync", "-avhP", '--remove-source-files'] + fileList + [destination, f'--bwlimit={bwLimit}']
        print(' '.join(command))
        try:
            process = subprocess.Popen(command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,bufsize=1)
            for line in process.stdout:
                if '% ' in line:
                    self.progress.clear_output()
                    with self.progress:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                else:
                    with self.output:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            process.wait()
            self.progress.clear_output()
            with self.output:
                print("Rsync completed successfully")
        except subprocess.CalledProcessError as e:
            with self.output:
                print(f"Rsync failed with error: {e.stderr.decode()}")
        except Exception as e:
            with self.output:
                print(f"An unexpected error occurred: {e}")
        self.update_files(None)