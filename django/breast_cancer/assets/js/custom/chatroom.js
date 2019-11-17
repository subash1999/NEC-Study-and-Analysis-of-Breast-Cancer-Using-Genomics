$(document).ready(function(){
	$('#other_user_menu_btn').click(function(){
		$('#other_user_menu').toggle();
	});
	$('#settings_menu_btn').click(function(){
		$('#settings_menu').toggle();
	});
	$(document).on("click", function(event){
		var $trigger = $("#other_user_menu_btn");
		if($trigger !== event.target && !$trigger.has(event.target).length){
			$("#other_user_menu").slideUp("fast");
		}    
		var $trigger = $("#settings_menu_btn");
		if($trigger !== event.target && !$trigger.has(event.target).length){
			$("#settings_menu").slideUp("fast");
		}            
	});
	// document.getElementsByClassName("user_img").onresize = function  () {
	// 	var wid = $(this).width();
	// 	var hei = $(this).height();
	// 	if(wid<height){
	// 		$(this).height(wid);
	// 	}
	// 	else{
	// 		$(this).width(hei);	
	// 	}
	// }

});
