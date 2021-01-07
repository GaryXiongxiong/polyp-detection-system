# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['C:\\Users\\Yixio\\OneDrive - Newcastle University\\Master Project\\PDS'],
             binaries=[(r'C:\Users\Yixio\anaconda3\envs\PDSdeploy\Lib\site-packages\tensorflow\lite\experimental\microfrontend\python\ops\_audio_microfrontend_op.so',r'.\tensorflow\lite\experimental\microfrontend\python\ops')],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn','tensorflow','tensorflow.python','tensorflow.python.keras','tensorflow.python.keras.engine.base_layer_v1'],
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
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=".",
          console=False,
          icon='data\\icon.ico')
