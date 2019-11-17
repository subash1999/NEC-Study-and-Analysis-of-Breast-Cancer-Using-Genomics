jQuery(document).ready(function($) {
	// USERNAME
	/********************************************************************************************
	********************************************************************************************
	********************************************************************************************/
	$('#edit_username_btn').click(function(event) {
		$('#username_label').hide();
		$(this).hide();

		$('#username').show();
		$('#cancel_username_btn').show();
		$('#save_username_btn').show();
	});
	
	$("#change_password_form").validate({
		rules: {
			password:{
				required : true,    
			} ,
			new_password:{
				required : true,    
				rangelength: [6, 255],	
			} ,
			new_password_confirmation:{
				required : true,    
				equalTo: "#new_password",	
			},

		},
		messages: {
			new_password: {
				required: "Please secure your account with password of min 6 chars",
			},
			new_password_confirmation: {
				required: "Please confirm the password",
				equalTo: "Password confrimation doesnot match with the above entered password",
			}
		},	
		errorClass: "is-invalid text-danger",
		validClass: "is-valid",
	});

	$("#change_username_form").validate({
		rules: {
			username: {
				required: true,
				rangelength : [2,50],
				remote: {
					url: check_username_available_url,
					type: "post",
					data: {
						"X-CSRFToken": getCookie("csrftoken"),
						"username" : console.log($('#username').val()),						
					},
				},
			},

		},
		messages: {
			username: {
				required : "You need an username to login",
				remote : 'Sorry !!! Username already taken. Try another on'
			},
		},	
		errorClass: "is-invalid text-danger",
		validClass: "is-valid",
	});

});