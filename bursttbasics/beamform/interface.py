import ipywidgets as widgets
import pandas as pd

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
        self.stationID.value = conf['General']['stationID']
        self.targetIP.value = conf['General']['transferIP']
        self.bwLimit.value = conf['General'].getint('bwLimit',1)
        self.saveDir.value = conf['General'].get('saveDir','./')