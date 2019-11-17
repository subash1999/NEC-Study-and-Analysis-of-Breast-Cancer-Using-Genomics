jQuery(document).ready(function($) {
  var page = 1;


  getConnectedUsers();

  $('#search_users').on('keyup',function(event) {
    page = 1;
    $('#nav_connected_users').html(`
     <div class="m-2" align="center" id="connected_users_loader">
     <label class="m-2">Loading Connected Users</label>
     ${loader}
     </div>
     `);
    getConnectedUsers();
  });
  function getConnectedUsers() {
    var search = $('#search_users').val();
    $.ajax({
      url: connected_users_search_url,
      type: 'GET',
      dataType: 'json',
      data: {'search': search,
      'page' : page,
    },
  })
    .done(function(datas) {
      showConnectedUsers(datas);
    })
    .fail(function() {
      showToastrMsg("error","Invalid Request !! Error Occured");
    })

  }


  function showConnectedUsers(datas) {
    var html = '';
    users = datas.users;
    if (users.length<1){
      html = `
      <div align="center" class="mt-5">
      <h4>No Result Found</h4>
      </div>
      `;
    }
    else{
      users.forEach(function (item,index) {
        html += `
        <div class="row m-2">
        <div class="img_cont_msg col-3">
        <img src="${item.profile_pic}" class="rounded-circle user_img img-fluid" title="${item.username}">
        </div>
        <div class="col-5">
        <div class="row">
        <label class="text-truncate col-10" title="${item.first_name} ${item.last_name}" style="font-weight:bold;" >${item.first_name} ${item.last_name}</label>
        </div>
        <div class="row">
        <label class="text-truncate col-10" title="${item.username}"><small>${item.username}</small></label>
        </div>
        </div>
        <div class="col-3">
        <form action="${new_msg_url}" method="POST">
        ${csrf_token}
        <input type="hidden" value="${item.id}" name="user_id" >
        <button class="btn btn-sm btn-info">Send Message</button>
        </form>
        </div>
        </div>
        <hr style="border-width:2px;">
        `;
      });
      if (datas.has_next_page){
        html+=`
        <div align="center" id="more_result_btn">
        <button type="button" class="btn btn-link" >More Result</button>
        </div>
        `;
      }
      else{
        html+=`
        <div align="center">
        <label>All Search Result Shown Above</label>
        </div>
        `;
      }
    }
    if (!datas.has_previous_page){
      $('#nav_connected_users').html(html);
    }
    else{
      $('#nav_connected_users').append(html);
    }

  }

  $('.tab-pane').on('click','#more_result_btn',function(event) {
    $(this).html('Loading Results ... ');
    page++;
    getConnectedUsers();
    $(this).remove();
  });

});