$(document).ready(function() {
	$("#login_form").validate({
		rules: {
			password:{
				required : true,    	
			} ,
			username: {
				required: true,
				
			}
		},
		messages: {
			username: {
				required: "Username Required !!! Username is what identifies you",
				
			}
		},
		errorClass: "is-invalid text-danger",
		validClass: "is-valid",
	});

	$('#register_now_btn').click(function(event) {
		$('.login').slideUp(800,function(){
			$('#login_form').trigger("reset");
			$('.register').slideDown(800);	
			document.title = 'Register'; 
		});
		
	});
});