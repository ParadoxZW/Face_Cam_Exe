# -*- mode: python -*-
import sys
sys.setrecursionlimit(5000)

block_cipher = None


a = Analysis(['camexe.py'],
             pathex=['C:\\Users\\96473\\Desktop\\cam'],
             binaries=[],
             datas=[('mxnet\\libgcc_s_seh-1.dll' ,'.'),
('mxnet\\libgfortran-3.dll' ,'.'),
('mxnet\\libmxnet.dll' ,'.'),
('mxnet\\libmxnet.lib' ,'libmxnet.lib'),
('mxnet\\libmxnet_static.lib' ,'.'),
('mxnet\\libquadmath-0.dll' ,'.'),
('mxnet\\proc.py' ,'.'),
('mxnet\\vcomp140.dll' ,'.'),
('C:\\Users\\96473\\Anaconda3\\Lib\\site-packages\\sklearn\\neighbors\\typedefs.cp36-win_amd64.pyd', 'typedefs.cp36-win_amd64.pyd')
],
             hiddenimports=['sklearn.neighbors.typedefs'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='camexe',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
