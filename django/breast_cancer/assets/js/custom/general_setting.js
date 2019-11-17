	
$(document).ready(function() {

	$('.general_setting_input').prop('disabled', true);
	$('#image-upload').hide();
	$('#image-label').hide();

	// upload preview image
	$.uploadPreview({
		input_field: "#profile_pic",
		preview_box: "#image-preview",
		label_field: "#image-label",
		initial_image : pp_url,
	});
	setBackgroundImage('image-preview');
	function setBackgroundImage(id) {
		$("#"+id).css("background-image", "url('"+pp_url+"')");
		$("#"+id).css("background-size", "cover");
		$("#"+id).css("background-position", "center center");
	}

	// edit button 
	$('#edit_general_setting_btn').click(function(event) {
		$(this).hide();

		$('.general_setting_input').prop('disabled', false);
		
		$('#image-upload').show();
		$('#image-label').show();
		$('#cancel_general_setting_btn').show();
		$('#save_general_setting_btn').show();
	});

	// validate the general setting form

	$.validator.addMethod('filesize', function (value, element, param) {
		return this.optional(element) || (element.files[0].size <= param)
	}, 'File size must be less than 1 Mb');
	
	jQuery.validator.addMethod("extension", function (value, element, param) {
		param = typeof param === "string" ? param.replace(/,/g, '|') : "png|jpe?g|gif";
		return this.optional(element) || value.match(new RegExp(".(" + param + ")$", "i"));
	}, "Please enter a value with a valid extension.");

	$("#general_setting_form").validate({
		rules: {
			profile_pic :{
				extension: "jpg|jpeg",
				filesize: 1024000,
			},
			first_name : {
				required : true,    
				rangelength: [2, 255],
			} ,
			last_name : {
				required : true,    
				rangelength: [2, 255],
			} ,
			gender : {
				rangelength : [0,10],
			},
			dob : {
				date : true,
			}
		},	
		errorPlacement: function(error, element) {
			if (element.attr("name") == "profile_pic") {
				error.insertAfter("#profile_pic_error");
			} else {
				error.insertAfter(element);
			}
		},
		errorClass: "is-invalid text-danger",
		validClass: "is-valid",
	});

});
