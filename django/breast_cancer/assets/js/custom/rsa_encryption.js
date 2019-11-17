
var RSA = {
	// you need to import the forge for this script to run

	rsa : {'public_key':null,'private_key':null,'pem_public_key':null,'pem_private_key':null,'keys':null},

	generateKeys : function (key_size){
		/*** Generating 1024-bit key-pair */  
		// defining the default value for the key_size if the value is not passed
		key_size = typeof key_size !== 'undefined' ? key_size : 1024;
		this.rsa.keys = forge.pki.rsa.generateKeyPair(1024);  
		// after generating the keys please set the private and public keys
		this.rsa.private_key = this.rsa.keys.privateKey;
		this.rsa.public_key = this.rsa.keys.publicKey;
		this.rsa.pem_public_key = forge.pki.publicKeyToPem(this.rsa.keys.publicKey)
		this.rsa.pem_private_key = forge.pki.privateKeyToPem(this.rsa.keys.privateKey)

	},

	getPrivateKey : function () {
		return this.rsa.private_key;
	},

	getPublicKey : function () {
		return this.rsa.public_key;
	},

	setPrivateKeyFromPEM : function (key) {
		this.rsa.pem_private_key = key;
		this.rsa.private_key = forge.pki.privateKeyFromPem(key);
		return true;
	},

	setPublicKeyFromPEM : function (key) {
		this.rsa.pem_public_key = key;
		this.rsa.public_key = forge.pki.publicKeyFromPem(key);
		return true;
	},

	encrypt : function (plaintext) {
		/*** public key encryption */ 
		if(this.rsa.public_key!=null){   
			// var publicKey = forge.pki.publicKeyFromPem(this.rsa.public_key);
			var publicKey = this.rsa.public_key
			var encrypted = publicKey.encrypt(plaintext, "RSA-OAEP", {
				md: forge.md.sha256.create(),
				mgf1: forge.mgf1.create()
			});
			var base64 = forge.util.encode64(encrypted);
			return base64;
		}
		else{
			console.log('Before Encryption please set Public Key');
			return false;
		}		
	},

	decrypt : function (cyphertext) {
		/*** public key encryption */ 
		if(this.rsa.private_key!=null){   
			// var privateKey = forge.pki.privateKeyFromPem(this.rsa.private_key);
			var privateKey = this.rsa.private_key
			var decoded64 = forge.util.decode64(cyphertext);			
			var decrypted = privateKey.decrypt(decoded64, "RSA-OAEP", {
				md: forge.md.sha256.create(),
				mgf1: forge.mgf1.create()
			});
			return decrypted;
		}
		else{
			console.log('Before Decryption please set Private Key');
			return false;
		}  
	},

	encryptFromPEM : function (plaintext) {
		/*** public key encryption */ 
		if(this.rsa.pem_public_key!=null){   
			var publicKey = forge.pki.publicKeyFromPem(this.rsa.pem_public_key);
			// var publicKey = this.rsa.public_key
			var encrypted = publicKey.encrypt(plaintext, "RSA-OAEP", {
				md: forge.md.sha256.create(),
				mgf1: forge.mgf1.create()
			});
			var base64 = forge.util.encode64(encrypted);
			return base64;
		}
		else{
			console.log('Before Encryption please set PEM Public Key i.e pem_public_key variable');
			return false;
		}		
	},

	decryptFromPEM : function (cyphertext) {
		/*** public key encryption */ 
		if(this.rsa.pem_private_key!=null){   
			var privateKey = forge.pki.privateKeyFromPem(this.rsa.pem_private_key);
			// var privateKey = this.rsa.private_key
			var decoded64 = forge.util.decode64(cyphertext);			
			var decrypted = privateKey.decrypt(decoded64, "RSA-OAEP", {
				md: forge.md.sha256.create(),
				mgf1: forge.mgf1.create()
			});
			return decrypted;
		}
		else{
			console.log('Before Decryption please set PEM Private Key i.e pem_private_key variable');
			return false;
		}  
	},
	
}
function PublicKey(public_key){
	this.public_key = public_key
	return this.public_key;
}