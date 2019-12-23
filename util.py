def write_to_excel(mech):
  """
  Excel writer
  """
  wb = openpyxl.load_workbook('./outputs/template/input.xlsx')
  ws = wb['Sheet1']

  img = openpyxl.drawing.image.Image(df.image_path[0])
  img.anchor = 'B5'
  img.width = 286
  img.height = 319
  ws.add_image(img)

  #ws.add_image(img, 'A1')
  wb.save('./outputs/output.xlsx')
  
  
def get_requirements():
  import pkg_resources
  import types
  
  def get_imports():
      for name, val in globals().items():
          if isinstance(val, types.ModuleType):
              # Split ensures you get root package, 
              # not just imported function
              name = val.__name__.split(".")[0]

          elif isinstance(val, type):
              name = val.__module__.split(".")[0]

          # Some packages are weird and have different
          # imported names vs. system/pip names. Unfortunately,
          # there is no systematic way to get pip names from
          # a package's imported name. You'll have to had
          # exceptions to this list manually!
          poorly_named_packages = {
              "PIL": "Pillow",
              "sklearn": "scikit-learn"
          }
          if name in poorly_named_packages.keys():
              name = poorly_named_packages[name]

          yield name
  imports = list(set(get_imports()))

  # The only way I found to get the version of the root package
  # from only the name of the package is to cross-check the names 
  # of installed packages vs. imported packages
  requirements = []
  for m in pkg_resources.working_set:
      if m.project_name in imports and m.project_name!="pip":
          requirements.append((m.project_name, m.version))

  for r in requirements:
      print("{}=={}".format(*r))
