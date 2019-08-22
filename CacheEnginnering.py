class CacheEnginnering(object):
	__slots__ = ('use_serialize', 'cache_path', 'base_forest')
	
	def __init__(self, base_forest):
		self.use_serialize = False
        self.cache_path = './__classifier__'
		self.base_forest = base_forest
	
	def __del__(self):
		del self.use_serialize
		del self.cache_path
		del self.base_forest
		
	# Com o modelo treinado salva os classificadores em disco
	def run(self):
		pass
		
	# Busca cada classificador individualmente e treina eles