$(document).ready(function() {
	$("#register_form").validate({
		groups: {
			fullname: "fname lname"
		},
		rules: {
			fname: {
				required: true,
				rangelength: [2, 255],
			},
			lname: {
				required: true,
				rangelength: [2, 255],
			},
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
			password:{
				required : true,    
				rangelength: [6, 255],	
			} ,
			password_confirmation:{
				required : true,    
				equalTo: "#password",	
			},

		},
		messages: {
			fname: {
				required : "Please Mention Both First and Last Name",
				rangelength : "No of Characters , Min : 2 and Max : 255 ",
			},
			lname: {
				required : "Please Mention Both First and Last Name",
				rangelength : "No of Characters , Min : 2 and Max : 255 ",
			},		
			username: {
				required : "You need an username to login",
				remote : 'Sorry !!! Username already taken. Try another on'
			},	
			password: {
				required: "Please secure your account with password of min 6 chars",
			},
			password_confirmation: {
				required: "Please confirm the password",
				equalTo: "Password confrimation doesnot match with the above entered password",
			}
		},
		errorPlacement: function(error, element) {
			if (element.attr("name") == "fname" || element.attr("name") == "lname" ) {
				error.insertAfter("#name_error");
			} else {
				error.insertAfter(element);
			}
		},
		errorClass: "is-invalid text-danger",
		validClass: "is-valid",
	});

	$('#login_now_btn').click(function(event) {

		$('.register').slideUp(800,function(){
			$('#register_form').trigger("reset");
			$('.login').slideDown(800);
			document.title = 'Login'; 	
		});		

	});
});