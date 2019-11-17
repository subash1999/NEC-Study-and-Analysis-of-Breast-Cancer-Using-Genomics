jQuery(document).ready(function($) {
	$('#online_status_checkbox').change(function() {
		var checked = $(this).prop('checked');
		var new_status = "Online";
		if (checked){
			new_status = "Online";
		}
		else{			
			new_status = "Offline";
		}
		$.ajax({
			url: update_status_url,
			type: 'POST',
			dataType: 'json',
			data: {'status': new_status},
		})
		.done(function(result) {
			showToastrMsg(result.msg_type,result.msg);
		})
		.fail(function() {
			showToastrMsg("error","Error while changing user chat status");
		})

	})
});